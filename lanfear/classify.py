"""Orbit classification from integrated + frequency-analysed orbits.

Following the standard picture (e.g. Carpintero & Aguilar 1998; Frigo et al.
2021), orbits are assigned to families from three cheap, already-reduced
per-orbit quantities:

* **Angular-momentum circulation.** A tube (loop) orbit circulates about one
  principal axis, so the corresponding component of angular momentum keeps its
  sign: ``circ_a = |<L_a>| / <|L_a|>`` is near 1. A box circulates about none
  (all ``circ_a`` small). The circulating axis names the tube: z -> short-axis
  tube, x -> long-axis tube, y -> (unstable) intermediate-axis tube. Long-axis
  tubes are *inner* or *outer* by the ratio of their long-axis to circulation
  frequency, ``|w_x| / sqrt(|w_y| |w_z|)`` (below ``inner_outer_ratio``: inner),
  calibrated on the exact families of a triaxial perfect ellipsoid. Because
  y-circulation is generically unstable, a candidate y-tube is corroborated
  against an x:z frequency lock (Frigo/Carpintero & Aguilar's own definition
  of a y-tube) and, failing that, the boxlet resonance test below: either a
  resonant box or an outright unconfirmed orbit can mimic persistent circ_y
  over a finite window without being a real tube.
* **Shape tensor.** Its smallest-to-largest eigenvalue ratio (``planarity``)
  vanishes for a planar orbit in any orientation; it is reported as a
  diagnostic, while rosettes are identified from the 1:1:1 frequency test.
* **Fundamental frequencies.** A low-order commensurability
  ``n_x w_x + n_y w_y + n_z w_z ~ 0`` among the axis fundamentals marks a
  resonant box (boxlet). Requires frequency data (:func:`analyse_family`).
  Boxlets are subdivided (:class:`BoxletType`): a two-axis resonance is
  named by its in-plane frequency ratio (banana 1:2, fish 2:3, pretzel 3:4;
  Miralda-Escudé & Schwarzschild 1989), with its plane and the axis it is
  elongated along; a three-axis resonance is *spatial*; and an orbit with two
  independent resonances is *closed* (a periodic orbit) rather than thin.
  The default tolerance ``1e-3`` was chosen with a null test (frequency
  triples with the axes shuffled independently between box orbits): at
  ``1e-2`` about 30% of such random triples have an order <= 5
  "resonance", against ~3% at ``1e-3``, where 70--95% of the boxlets found in
  test models are genuine. Orders 6 and 7 are dominated by chance hits, so the
  search stops at order 5, except for the named pretzel (order 7), which is
  searched for explicitly.
* **Frequency diffusion.** An orbit whose leading frequencies drift by more
  than ``diffusion_threshold`` (Laskar) between the two halves of the
  integration is labelled irregular (chaotic). If it was re-integrated for
  longer, its drift must also fail to fall with the longer window.
* **Number of base frequencies.** An orbit whose spectrum needs more than three
  independent base frequencies is *irregular* (Frigo et al. 2021) -- it is not
  confined to a regular 3-torus and is a likely-chaotic candidate. This label
  overrides the regular-family assignment and requires frequency data.

The classification runs on the compact arrays in an :class:`OrbitResults`, so it
is trivially fast even for millions of orbits.

For a rotating figure (a non-zero :attr:`OrbitResults.pattern_speed`), every
one of these quantities -- angular momentum, shape tensor, frequencies -- is
measured in the co-rotating frame of the figure, where the potential is static
and each regular orbit is a steady torus (see :mod:`lanfear.orbits`). The
tests above are therefore applied unchanged, with resonances meaning
resonances with the pattern. Three things matter from a static potential:

* **Precessing boxes.** In a tumbling figure a box precesses steadily about
  the rotation axis, so it winds about that axis and its angular momentum
  there keeps one sign most of the time. It still passes close to the centre,
  so a loop about the pattern-speed axis also needs a hole about it
  (``hole_thresh``, :meth:`OrbitResults.hole_fraction`); the tumbling perfect
  ellipsoids of :doc:`tuning` show both the hole and the high ``circ_thresh``
  are needed.
* **Rosettes** are tested on frame-corrected frequencies: for a loop about the
  pattern-speed axis the two in-plane fundamentals are replaced by its
  inertial azimuthal frequency (:meth:`OrbitResults.azimuthal_frequency`).
* **Pattern-locked orbits** (:meth:`OrbitResults.pattern_locked`) barely move
  in the inertial frame, so in the co-rotating frame their x and y share the
  frequency ``|Omega_p|``. That lock is the frame rotation, so such an orbit is
  never labelled a rosette.

The thresholds were calibrated against the exact families of equilibrium
perfect ellipsoids, both static and tumbling about the short axis with
corotation at 16, 8 and 4 half-mass radii (see :doc:`tuning`). In the slowly
tumbling ones (16 and 8 half-mass radii) the families are as reliable as in a
static figure (92--94% right; 96% of the regular orbits); with corotation at 4
half-mass radii, or for orbits flagged
:attr:`~OrbitClassification.near_corotation`, treat them with care. Fast,
bar-like figures have been checked only against pinned orbits, not an exact
reference.

Every loop also gets a *sense* (:class:`OrbitSense`): prograde or
retrograde about its circulation axis, from the sign of its azimuthal
frequency (:meth:`OrbitResults.azimuthal_frequency`), taken in the inertial
frame about the pattern-speed axis. What "prograde" refers to is set by the
axis signs, which :meth:`lanfear.ParticleSystem.orient_axes` fixes so that +z
points along the pattern speed (or, for a static figure, along the net spin)
and +x along the net ``L_x`` (see :attr:`OrbitResults.orientation`). In a
rotating figure, pattern-locked orbits are ``LOCKED``, and non-loops that keep
clear of the pattern-speed axis without going round it are ``LIBRATING``
(e.g. orbits trapped around the Lagrange points L4/L5 -- or a resonant box
that avoids the centre: librating alone is not corotation trapping).

In a rotating figure, orbits that go round the pattern-speed axis (loops about
it, and librating orbits) also get a *pattern resonance*
(:class:`PatternResonance`), from the condition
``l Omega_R + m Omega_phi,body = 0`` between the radial frequency and the
body-frame azimuthal frequency ``Omega_phi,body = Omega_phi - Omega_p``,
accepted within ``pattern_tol`` frequency resolutions:

* corotation (``Omega_phi,body = 0``), provided the pattern speed itself
  completes ``pattern_cycles`` cycles in the window (otherwise corotation
  cannot be told from no rotation, and the orbit is ``UNRESOLVED``); a
  librating orbit at corotation is trapped around L4/L5
  (:attr:`OrbitClassification.corotation_trapped`);
* 2:1 (``Omega_R = 2 |Omega_phi,body|``): the families of closed loops of a
  bar-like (m = 2) figure, whose radius oscillates twice per turn in the
  co-rotating frame. Prograde in the body frame, they are ``X1`` (elongated
  along the long axis, the bar-supporting family) or ``X2`` (across it);
  retrograde, they are ``X4`` if also retrograde in the inertial frame and at
  the outer Lindblad resonance (``OLR``) otherwise;
* 4:1 ultraharmonic (``Omega_R = 4 |Omega_phi,body|``).

Orbits whose radial or azimuthal frequency completes fewer than
``pattern_cycles`` cycles in the window are ``UNRESOLVED`` rather than guessed.
On rotating test models the ratio ``Omega_R / |Omega_phi,body|`` has a sharp
spike at exactly 2 (the trapped orbits) above a broad continuum, nearly all of
it within half a frequency resolution.

Separately from the family label, every orbit gets a *regularity*
(:class:`Regularity`: regular, irregular by its spectrum, or diffusing) and
two reliability flags: :attr:`OrbitClassification.unresolved` (a frequency one
of its labels depends on completes too few cycles in the window) and, in a
rotating figure, :attr:`OrbitClassification.near_corotation` (its own pattern
speed ratio ``epsilon = |Omega_p| / |Omega_phi|`` lies in
``corotation_band``, by default at or beyond corotation, ``epsilon >= 0.5``,
where the classification is least reliable).

The separate labels are combined into one named *family* per orbit by an
ordered rule table (:data:`FAMILY_TABLE`, first match wins, ``"other"`` if no
rule matches; :attr:`OrbitClassification.family`): e.g. ``"x1"``,
``"banana"``, ``"prograde_short_axis_tube"`` or ``"corotation_trapped"``.
Adding a family is a new :class:`FamilyRule`, not new classification logic.
The original family label (:attr:`OrbitClassification.labels`,
:class:`OrbitClass`) is derived from the shape, boxlet type and regularity, so
existing code keeps working; most methods take a ``scheme`` argument to work on
any of the labels instead (see :meth:`OrbitClassification.view`).

Every aggregate over orbits (counts, binned fractions, histograms, Sankey
transitions) can be **mass-weighted**: pass the
:class:`~lanfear.ParticleSystem` the orbits were drawn from as ``particles``,
and each orbit counts with the mass of the particle with its ID
(:func:`particle_weights`) instead of once. Masses are not stored with the
orbits, so the same classification can be weighted by any particle set that
contains its IDs.

Besides the family label, every orbit gets a purely geometric *shape*
(:class:`OrbitShape`: box, one of the tubes, or rosette), assigned before the
irregular override and without the boxlet split, so a chaotic orbit keeps the
shape it has over the integration window.

Orbits that leave the system within the integration
(:attr:`OrbitResults.runaway`, typically stars beyond corotation of a rotating
figure) would get meaningless labels, so :func:`classify_orbits` leaves them out
by default; the classification then covers fewer orbits than the results, and
is matched to them by particle ID. ``drop_runaways=False`` keeps them (with a
warning).

Orbits whose pericentre reaches a central SMBH binary
(:attr:`OrbitResults.binary_interacting`) are classified like any other by
default; pass ``drop_binary_interacting=True`` to :func:`classify_orbits` to
leave them out, since no static potential represents their scattering by the
binary.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import IntEnum
from math import gcd
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from ._logging import get_logger

logger = get_logger(__name__)


class OrbitClass(IntEnum):
    """Enumeration of orbit families assigned by :func:`classify_orbits`."""

    UNCLASSIFIED = 0
    PIBOX = 1
    BOXLET = 2  # resonant box (banana/fish/pretzel/...)
    INTERMEDIATE_AXIS_TUBE = 3  # y-tube (generally unstable)
    SHORT_AXIS_TUBE = 4  # z-tube
    INNER_LONG_AXIS_TUBE = 5  # inner x-tube
    OUTER_LONG_AXIS_TUBE = 6  # outer x-tube
    ROSETTE = 7  # planar loop
    IRREGULAR = 8  # > 3 base frequencies (Frigo et al. 2021) -- likely chaotic


CLASS_NAMES = {c.value: c.name.lower() for c in OrbitClass}


class OrbitFamily(IntEnum):
    """Condensed orbit families (box vs tube).

    Produced by :meth:`OrbitClassification.condense_families`, which groups the
    detailed :class:`OrbitClass` subclasses into the fundamental box/tube
    dichotomy.
    """

    UNCLASSIFIED = 0
    BOX = 1
    TUBE = 2


CONDENSED_NAMES = {f.value: f.name.lower() for f in OrbitFamily}


class OrbitShape(IntEnum):
    """Geometric orbit shape assigned by :func:`classify_orbits`.

    The family label without the boxlet/π-box split and without the
    irregular override (see :attr:`OrbitClassification.shape`): what the
    orbit looks like over the integration window, whether or not it is
    regular.
    """

    UNCLASSIFIED = 0  # integration failed
    BOX = 1
    SHORT_AXIS_TUBE = 2
    INNER_LONG_AXIS_TUBE = 3
    OUTER_LONG_AXIS_TUBE = 4
    INTERMEDIATE_AXIS_TUBE = 5
    ROSETTE = 6


SHAPE_NAMES = {s.value: s.name.lower() for s in OrbitShape}

_LOOP_SHAPES = (
    OrbitShape.SHORT_AXIS_TUBE,
    OrbitShape.INNER_LONG_AXIS_TUBE,
    OrbitShape.OUTER_LONG_AXIS_TUBE,
    OrbitShape.INTERMEDIATE_AXIS_TUBE,
    OrbitShape.ROSETTE,
)


class OrbitSense(IntEnum):
    """Sense of motion assigned by :func:`classify_orbits`.

    ``PROGRADE``/``RETROGRADE`` apply to loops (tubes and rosettes), about
    their circulation axis and relative to the axis signs fixed by
    :meth:`lanfear.ParticleSystem.orient_axes`. ``LOCKED`` and ``LIBRATING``
    occur only in rotating figures. Boxes (and failed orbits) have ``NONE``.
    """

    NONE = 0
    PROGRADE = 1
    RETROGRADE = 2
    LOCKED = 3  # barely moves in the inertial frame
    LIBRATING = 4  # avoids the pattern axis without going round it


SENSE_NAMES = {s.value: s.name.lower() for s in OrbitSense}


class PatternResonance(IntEnum):
    """Resonance with a rotating figure assigned by :func:`classify_orbits`.

    See the module notes. ``NONE`` for orbits that are not resonant, do not go
    round the pattern-speed axis, or belong to a static figure.
    """

    NONE = 0
    UNRESOLVED = 1  # frequencies too slow for the window
    COROTATION = 2  # Omega_phi = Omega_p
    X1 = 3  # 2:1, prograde in the body frame, along the long axis
    X2 = 4  # 2:1, prograde in the body frame, across the long axis
    X4 = 5  # 2:1, retrograde in both frames
    OLR = 6  # 2:1, retrograde in the body frame, prograde inertially
    ULTRAHARMONIC = 7  # 4:1


PATTERN_RESONANCE_NAMES = {r.value: r.name.lower() for r in PatternResonance}


class Regularity(IntEnum):
    """Regularity assigned by :func:`classify_orbits`.

    ``IRREGULAR``: the spectrum needs more than three base frequencies (Frigo
    et al. 2021). ``DIFFUSING``: the frequencies drift between the two halves
    of the integration by more than ``diffusion_threshold`` (Laskar); it takes
    precedence when both apply. Either overrides the family label with
    :attr:`OrbitClass.IRREGULAR`.
    """

    UNKNOWN = 0  # integration failed
    REGULAR = 1
    IRREGULAR = 2  # > 3 base frequencies
    DIFFUSING = 3  # frequency diffusion above the threshold


REGULARITY_NAMES = {r.value: r.name.lower() for r in Regularity}

# Range of epsilon = |Omega_p| / |Omega_phi| (OrbitResults.pattern_speed_ratio)
# flagged as near corotation. On tumbling perfect ellipsoids the labels of
# orbits with epsilon < 0.5 agree with the reference as often as in a static
# figure (95%), those at and beyond corotation far less (0.4-0.8), with no
# recovery at large epsilon, so the band is open-ended.
DEFAULT_COROTATION_BAND = (0.5, np.inf)

# Inner/outer long-axis tubes: fallback cut on |w_x| / sqrt(|w_y| |w_z|) when no
# adaptive cut can be placed (calibrated on a triaxial perfect ellipsoid with
# b/a = 0.79, c/a = 0.5; the best cut ranges 0.79--0.90 from prolate to
# near-oblate shapes).
DEFAULT_INNER_OUTER_RATIO = 0.84
# Acceptance tests of the adaptive (Otsu) cut, set on four perfect ellipsoids
# (prolate to near-oblate), where they accept every mixed population and
# reject every single-family one: enough long-axis tubes; the cut within the
# range of best cuts; an outer peak (upper-group mean near the universal outer
# value ~0.95); the two groups well apart; and neither group negligible.
_ADAPTIVE_MIN_TUBES = 200
_ADAPTIVE_CUT_RANGE = (0.75, 0.91)
_ADAPTIVE_MIN_OUTER_MEAN = 0.92
_ADAPTIVE_MIN_GAP = 0.10
_ADAPTIVE_FRACTION_RANGE = (0.1, 0.9)


def adaptive_inner_outer_cut(ratio):
    """Cut between inner and outer long-axis tubes from the population itself.

    Inner and outer long-axis tubes differ in ``|w_x| / sqrt(|w_y| |w_z|)``:
    outer tubes sit at ~0.94--0.95 for every figure shape, inner tubes lower,
    by an amount that depends on the shape (~0.67 for a prolate to ~0.82 for a
    near-oblate perfect ellipsoid). The cut is placed by Otsu's method (the
    split maximising the between-group variance) on the ratios of the
    population's long-axis tubes; on four perfect ellipsoids it lands within
    ~0.01 of the best possible cut in balanced accuracy. It is rejected when
    the population does not look like a mix of the two families (see the
    ``_ADAPTIVE_*`` acceptance constants).

    Parameters
    ----------
    ratio : numpy.ndarray
        ``|w_x| / sqrt(|w_y| |w_z|)`` of the long-axis tubes (non-finite values
        are ignored).

    Returns
    -------
    cut : float or None
        The cut, or None if it was rejected.
    reason : str
        Why it was rejected (empty if accepted).
    """
    values = np.asarray(ratio, dtype=np.float64)
    values = np.clip(values[np.isfinite(values)], 0.4, 1.1)
    if len(values) < _ADAPTIVE_MIN_TUBES:
        return None, f"only {len(values)} long-axis tubes (< {_ADAPTIVE_MIN_TUBES})"
    counts, edges = np.histogram(values, bins=140, range=(0.4, 1.1))
    centres = 0.5 * (edges[1:] + edges[:-1])
    below = np.cumsum(counts)
    above = below[-1] - below
    sum_below = np.cumsum(counts * centres)
    mean_below = sum_below / np.maximum(below, 1)
    mean_above = (sum_below[-1] - sum_below) / np.maximum(above, 1)
    k = int(np.argmax(below * above * (mean_below - mean_above) ** 2))
    cut = float(edges[k + 1])
    low, high = values[values < cut], values[values >= cut]
    fraction = len(low) / len(values)
    if not _ADAPTIVE_CUT_RANGE[0] <= cut <= _ADAPTIVE_CUT_RANGE[1]:
        return None, f"Otsu cut {cut:.3f} outside {_ADAPTIVE_CUT_RANGE}"
    if not _ADAPTIVE_FRACTION_RANGE[0] <= fraction <= _ADAPTIVE_FRACTION_RANGE[1]:
        return None, f"{fraction:.0%} of the tubes below the cut"
    if high.mean() < _ADAPTIVE_MIN_OUTER_MEAN:
        return None, f"no outer peak (upper mean {high.mean():.3f})"
    if high.mean() - low.mean() < _ADAPTIVE_MIN_GAP:
        return None, f"groups too close ({low.mean():.3f} vs {high.mean():.3f})"
    return cut, ""


# Shape of each regular family label (before the irregular override).
_CLASS_TO_SHAPE = {
    OrbitClass.UNCLASSIFIED: OrbitShape.UNCLASSIFIED,
    OrbitClass.PIBOX: OrbitShape.BOX,
    OrbitClass.BOXLET: OrbitShape.BOX,
    OrbitClass.SHORT_AXIS_TUBE: OrbitShape.SHORT_AXIS_TUBE,
    OrbitClass.INNER_LONG_AXIS_TUBE: OrbitShape.INNER_LONG_AXIS_TUBE,
    OrbitClass.OUTER_LONG_AXIS_TUBE: OrbitShape.OUTER_LONG_AXIS_TUBE,
    OrbitClass.INTERMEDIATE_AXIS_TUBE: OrbitShape.INTERMEDIATE_AXIS_TUBE,
    OrbitClass.ROSETTE: OrbitShape.ROSETTE,
}

# Which detailed OrbitClass subclasses fold into each condensed OrbitFamily.
_TUBE_CLASSES = (
    OrbitClass.SHORT_AXIS_TUBE,
    OrbitClass.INNER_LONG_AXIS_TUBE,
    OrbitClass.OUTER_LONG_AXIS_TUBE,
    OrbitClass.INTERMEDIATE_AXIS_TUBE,
    OrbitClass.ROSETTE,  # a rosette is a (planar) loop orbit
)
_BOX_CLASSES = (OrbitClass.PIBOX, OrbitClass.BOXLET)


class BoxletType(IntEnum):
    """Subdivision of the boxlets (see :attr:`OrbitClassification.boxlet_type`).

    A two-axis resonance ``n_i w_i + n_j w_j = 0`` is named by its in-plane
    frequency ratio, lower:higher (Miralda-Escudé & Schwarzschild 1989); a
    resonance involving all three axes is spatial.
    """

    NONE = 0  # not a boxlet
    BANANA = 1  # 1:2
    FISH = 2  # 2:3
    PRETZEL = 3  # 3:4
    PLANAR = 4  # any other two-axis resonance (e.g. 1:1, 1:3)
    SPATIAL = 5  # a three-axis resonance, e.g. (1, -2, 1)


BOXLET_TYPE_NAMES = {b.value: b.name.lower() for b in BoxletType}

# Named two-axis resonances: in-plane frequency ratio (lower, higher) -> type.
NAMED_BOXLETS = {
    (1, 2): BoxletType.BANANA,
    (2, 3): BoxletType.FISH,
    (3, 4): BoxletType.PRETZEL,
}


@dataclass(frozen=True)
class FamilyRule:
    """One rule of a family table (see :data:`FAMILY_TABLE`).

    An orbit matches when every given condition holds: its label of that kind
    is one of the listed values. ``None`` (the default) accepts any value.

    Parameters
    ----------
    name : str
        Family name assigned to matching orbits (unique within a table).
    shape : tuple of OrbitShape, optional
    boxlet : tuple of BoxletType, optional
    sense : tuple of OrbitSense, optional
    resonance : tuple of PatternResonance, optional
    regularity : tuple of Regularity, optional
    """

    name: str
    shape: Optional[tuple] = None
    boxlet: Optional[tuple] = None
    sense: Optional[tuple] = None
    resonance: Optional[tuple] = None
    regularity: Optional[tuple] = None

    def matches(self, arrays: Dict[str, np.ndarray]) -> np.ndarray:
        """Which orbits satisfy every condition of this rule.

        Parameters
        ----------
        arrays : dict
            ``"shape"``, ``"boxlet"``, ``"sense"``, ``"resonance"`` and
            ``"regularity"`` -> (N,) integer label arrays.

        Returns
        -------
        mask : numpy.ndarray
            (N,) bool.
        """
        mask = np.ones(len(arrays["shape"]), dtype=bool)
        for kind in ("shape", "boxlet", "sense", "resonance", "regularity"):
            allowed = getattr(self, kind)
            if allowed is not None:
                mask &= np.isin(arrays[kind], [int(v) for v in allowed])
        return mask


def _tube_rules():
    """Tube and rosette families, split by sense (then without a sense)."""
    loops = (
        OrbitShape.SHORT_AXIS_TUBE,
        OrbitShape.INNER_LONG_AXIS_TUBE,
        OrbitShape.OUTER_LONG_AXIS_TUBE,
        OrbitShape.INTERMEDIATE_AXIS_TUBE,
        OrbitShape.ROSETTE,
    )
    rules = [
        FamilyRule(
            f"{SENSE_NAMES[int(sense)]}_{SHAPE_NAMES[int(shape)]}",
            shape=(shape,),
            sense=(sense,),
        )
        for shape in loops
        for sense in (OrbitSense.PROGRADE, OrbitSense.RETROGRADE)
    ]
    rules += [FamilyRule(SHAPE_NAMES[int(shape)], shape=(shape,)) for shape in loops]
    return tuple(rules)


# The default family table, in priority order (first match wins). Failed and
# irregular orbits first, then the rotating-figure families (which override
# the shape they would otherwise have), then boxlets, librating orbits, plain
# boxes, and the tubes and rosettes by sense.
FAMILY_TABLE = (
    FamilyRule("unclassified", shape=(OrbitShape.UNCLASSIFIED,)),
    FamilyRule("irregular", regularity=(Regularity.IRREGULAR, Regularity.DIFFUSING)),
    FamilyRule("pattern_locked", sense=(OrbitSense.LOCKED,)),
    FamilyRule(
        "corotation_trapped",
        sense=(OrbitSense.LIBRATING,),
        resonance=(PatternResonance.COROTATION,),
    ),
    FamilyRule("x1", resonance=(PatternResonance.X1,)),
    FamilyRule("x2", resonance=(PatternResonance.X2,)),
    FamilyRule("x4", resonance=(PatternResonance.X4,)),
    FamilyRule("olr", resonance=(PatternResonance.OLR,)),
    FamilyRule("ultraharmonic", resonance=(PatternResonance.ULTRAHARMONIC,)),
    FamilyRule("banana", boxlet=(BoxletType.BANANA,)),
    FamilyRule("fish", boxlet=(BoxletType.FISH,)),
    FamilyRule("pretzel", boxlet=(BoxletType.PRETZEL,)),
    FamilyRule("planar_boxlet", boxlet=(BoxletType.PLANAR,)),
    FamilyRule("spatial_boxlet", boxlet=(BoxletType.SPATIAL,)),
    FamilyRule("librating", sense=(OrbitSense.LIBRATING,)),
    FamilyRule("pibox", shape=(OrbitShape.BOX,)),
) + _tube_rules()

# Combinations of labels that should never occur together. Most are excluded
# by construction; the checks guard against regressions and hand-edited
# classifications. Each predicate takes an OrbitClassification.
INCONSISTENCY_CHECKS = (
    (
        "loop at corotation but retrograde",
        lambda c: (c.pattern_resonance == PatternResonance.COROTATION)
        & (c.sense == OrbitSense.RETROGRADE),
    ),
    (
        "x1/x2 (prograde in the body frame) but retrograde",
        lambda c: np.isin(
            c.pattern_resonance, [PatternResonance.X1, PatternResonance.X2]
        )
        & (c.sense == OrbitSense.RETROGRADE),
    ),
    (
        "x4 but prograde",
        lambda c: (c.pattern_resonance == PatternResonance.X4)
        & (c.sense == OrbitSense.PROGRADE),
    ),
    (
        "pattern-locked with a pattern resonance",
        lambda c: (c.sense == OrbitSense.LOCKED)
        & (c.pattern_resonance != PatternResonance.NONE),
    ),
    (
        "pattern resonance, locking or libration in a static figure",
        lambda c: _static(c)
        & (
            (c.pattern_resonance != PatternResonance.NONE)
            | np.isin(c.sense, [OrbitSense.LOCKED, OrbitSense.LIBRATING])
        ),
    ),
    (
        "boxlet type on an orbit that is not a box",
        lambda c: (c.boxlet_type != BoxletType.NONE) & (c.shape != OrbitShape.BOX),
    ),
    (
        "prograde/retrograde sense on a box",
        lambda c: (c.shape == OrbitShape.BOX)
        & np.isin(c.sense, [OrbitSense.PROGRADE, OrbitSense.RETROGRADE]),
    ),
)


def _static(classification) -> np.ndarray:
    """Per-orbit mask: True everywhere if the classified figure is static."""
    ratio = classification.pattern_speed_ratio
    static = ratio is None or not np.any(np.nan_to_num(ratio))
    return np.full(len(classification.labels), static)


def _legacy_labels(shape, boxlet_type, regularity) -> np.ndarray:
    """The :class:`OrbitClass` family label implied by the separate labels.

    Parameters
    ----------
    shape, boxlet_type, regularity : numpy.ndarray
        (N,) :class:`OrbitShape`, :class:`BoxletType` and :class:`Regularity`
        values.

    Returns
    -------
    labels : numpy.ndarray
        (N,) :class:`OrbitClass` values: irregular orbits are ``IRREGULAR``,
        boxes ``BOXLET`` if they have a boxlet type and ``PIBOX`` otherwise, and
        the other shapes map to the class of the same name.
    """
    labels = np.full(len(shape), OrbitClass.UNCLASSIFIED, dtype=np.int64)
    for cls, shp in _CLASS_TO_SHAPE.items():
        if shp not in (OrbitShape.BOX, OrbitShape.UNCLASSIFIED):
            labels[shape == shp] = cls
    box = shape == OrbitShape.BOX
    labels[box] = OrbitClass.PIBOX
    labels[box & (boxlet_type != BoxletType.NONE)] = OrbitClass.BOXLET
    labels[np.isin(regularity, [Regularity.IRREGULAR, Regularity.DIFFUSING])] = (
        OrbitClass.IRREGULAR
    )
    return labels


def assign_families(arrays: Dict[str, np.ndarray], table=FAMILY_TABLE):
    """Name each orbit's family from its labels with an ordered rule table.

    Parameters
    ----------
    arrays : dict
        ``"shape"``, ``"boxlet"``, ``"sense"``, ``"resonance"`` and
        ``"regularity"`` -> (N,) label arrays.
    table : sequence of FamilyRule, optional
        Rules in priority order (default :data:`FAMILY_TABLE`).

    Returns
    -------
    family : numpy.ndarray
        (N,) index of the first matching rule, or ``len(table)`` for none.
    names : dict
        Index -> family name, with ``len(table)`` -> ``"other"``.

    Raises
    ------
    ValueError
        If two rules share a name or one is called ``"other"``.
    """
    names = [rule.name for rule in table]
    if len(set(names)) != len(names) or "other" in names:
        raise ValueError("family rule names must be unique and not 'other'")
    n = len(arrays["shape"])
    family = np.full(n, len(table), dtype=np.int64)
    open_ = np.ones(n, dtype=bool)
    for k, rule in enumerate(table):
        hit = open_ & rule.matches(arrays)
        family[hit] = k
        open_ &= ~hit
    return family, {**dict(enumerate(names)), len(table): "other"}


# LaTeX labels used when rendering family names on plot legends and axes (see
# the OrbitClassification.plot_* methods). Keyed by the family name as it appears
# in CLASS_NAMES / CONDENSED_NAMES; values are raw strings holding the LaTeX to
# render (e.g. r"$x$-tube"). FILL THESE IN -- an empty entry falls back to the
# plain family name via _latex_label(), so plots stay legible until you do.
LATEX_LABELS = {
    "unclassified": "unclassified",
    "pibox": r"$\pi$-box",
    "boxlet": "$boxlet",
    "intermediate_axis_tube": r"$y$-tube",
    "short_axis_tube": r"$z$-tube",
    "inner_long_axis_tube": r"inner $x$-tube",
    "outer_long_axis_tube": r"outer $x$-tube",
    "rosette": "rosette",
    "irregular": "irregular",
    "banana": "banana",
    "fish": "fish",
    "pretzel": "pretzel",
    "planar": "planar",
    "spatial": "spatial",
    "tube": "tube",  # (condensed family)
    "box": "box",  # (condensed family)
}


def _latex_label(name: str) -> str:
    """LaTeX label for a family name, falling back to the plain name.

    Parameters
    ----------
    name : str
        Family name as stored in :data:`CLASS_NAMES` / :data:`CONDENSED_NAMES`.

    Returns
    -------
    label : str
        The corresponding entry in :data:`LATEX_LABELS` if it is non-empty,
        otherwise ``name`` unchanged.
    """
    return LATEX_LABELS.get(name) or name


# Default colour palette used for orbit-class plots when no other palette is
# requested. Keyed by the family name as it appears in CLASS_NAMES (mirrors
# LATEX_LABELS above).
DEFAULT_PALETTE = {
    "unclassified": "#999999",
    "pibox": "#9C0B10",
    "boxlet": "#d34936",
    "intermediate_axis_tube": "#85C8E1",  # y-tube
    "short_axis_tube": "#3885d0",  # z-tube
    "inner_long_axis_tube": "#56247C",  # inner x-tube
    "outer_long_axis_tube": "#7B5AAB",  # outer x-tube
    "rosette": "#112C50",
    "irregular": "#797878FF",
    "tube": "#112C50",  # condensed family
    "box": "#9C0B10",  # condensed family
}


def _colour_for(name: str, index: int = 0) -> str:
    """Colour for a family name from :data:`DEFAULT_PALETTE`.

    Parameters
    ----------
    name : str
        Family name as stored in :data:`CLASS_NAMES` / :data:`CONDENSED_NAMES`.
    index : int, optional
        Position of ``name`` among the families being plotted, used to pick a
        colour from matplotlib's default cycle when ``name`` has no entry in
        :data:`DEFAULT_PALETTE` (e.g. the condensed "box"/"tube" families).

    Returns
    -------
    colour : str
        A matplotlib-compatible colour spec.
    """
    if name in DEFAULT_PALETTE:
        return DEFAULT_PALETTE[name]
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    return cycle[index % len(cycle)]


def _format_total(value) -> str:
    """A node total for a plot label: an integer count, or a mass to 3 digits."""
    value = float(value)
    return f"{int(value)}" if value.is_integer() else f"{value:.3g}"


def particle_weights(ids, particles=None) -> np.ndarray:
    """Mass of the particle behind each classified orbit, matched by ID.

    Parameters
    ----------
    ids : array-like
        (N,) particle IDs of the classified orbits.
    particles : lanfear.ParticleSystem, optional
        A particle system containing every ID in ``ids`` (e.g. the one the
        orbits were integrated from; extra particles are ignored). None gives
        every orbit unit weight.

    Returns
    -------
    weights : numpy.ndarray
        (N,) particle masses (physical units of ``particles``), or ones.

    Raises
    ------
    ValueError
        If ``ids`` is None while ``particles`` is given, if ``particles`` has
        duplicate IDs, or if any ID is missing from it.
    """
    if particles is None:
        return np.ones(0 if ids is None else len(ids))
    if ids is None:
        raise ValueError(
            "mass weighting needs the particle IDs of the classified orbits"
        )
    ids = np.asarray(ids)
    particle_ids = np.asarray(particles.ids)
    order = np.argsort(particle_ids, kind="stable")
    sorted_ids = particle_ids[order]
    if np.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise ValueError("the particle system has duplicate IDs; cannot match by ID")
    position = np.searchsorted(sorted_ids, ids)
    clipped = np.minimum(position, len(sorted_ids) - 1)
    found = (position < len(sorted_ids)) & (sorted_ids[clipped] == ids)
    if not found.all():
        raise ValueError(
            f"{int(np.sum(~found))} of {len(ids)} classified orbits have no particle "
            f"with their ID in the particle system"
        )
    return np.asarray(particles.mass, dtype=np.float64)[order[clipped]]


@dataclass
class ClassFractions:
    """Class fractions in bins of a per-orbit quantity (from :meth:`OrbitClassification.fractions_by`).

    Attributes
    ----------
    edges : numpy.ndarray
        (n_bins + 1,) bin edges.
    centres : numpy.ndarray
        (n_bins,) bin centres.
    labels : numpy.ndarray
        (n_classes,) integer labels of the classes present.
    class_names : dict
        Mapping from integer label to name.
    counts : numpy.ndarray
        (n_classes, n_bins) number of orbits per class and bin, or their total
        mass if :attr:`weighted`.
    fractions : numpy.ndarray
        (n_classes, n_bins) class fractions (NaN in empty bins if per-bin).
    lower, upper : numpy.ndarray or None
        (n_classes, n_bins) bootstrap confidence bounds, ``None`` if no
        bootstrap was requested.
    per_bin : bool
        Whether fractions are normalised within each bin or to the total.
    quantity_label : str
        Axis label describing the binned quantity.
    weighted : bool
        Whether orbits were weighted by their particle mass.
    """

    edges: np.ndarray
    centres: np.ndarray
    labels: np.ndarray
    class_names: Dict[int, str]
    counts: np.ndarray
    fractions: np.ndarray
    lower: Optional[np.ndarray]
    upper: Optional[np.ndarray]
    per_bin: bool
    quantity_label: str
    weighted: bool = False

    def plot(self, ax=None, **kwargs):
        """Draw one curve per class, with the bootstrap band if available.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. A new figure and axes are created if omitted.
        **kwargs
            Passed through to ``ax.plot`` for every class.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes the curves were drawn on.
        """
        if ax is None:
            _, ax = plt.subplots()
        base_kwargs = dict(kwargs)
        base_kwargs.setdefault("ls", "-")
        base_kwargs.setdefault("lw", 2)
        for i, cls in enumerate(self.labels):
            name = self.class_names[int(cls)]
            plot_kwargs = dict(base_kwargs)
            plot_kwargs.setdefault("color", _colour_for(name, i))
            (line,) = ax.plot(
                self.centres, self.fractions[i], label=_latex_label(name), **plot_kwargs
            )
            if self.lower is not None:
                ax.fill_between(
                    self.centres,
                    self.lower[i],
                    self.upper[i],
                    color=line.get_color(),
                    alpha=0.2,
                    lw=0,
                )
        ax.set_xlabel(self.quantity_label)
        kind = "mass fraction" if self.weighted else "fraction"
        ax.set_ylabel(f"{kind} within bin" if self.per_bin else f"{kind} of all orbits")
        ax.figure.legend(loc="outside upper center", ncols=3, fontsize="small")
        ax.figure.subplots_adjust(top=0.85)
        return ax


@dataclass
class OrbitClassification:
    """Result of :func:`classify_orbits` (all arrays indexed like the orbits).

    Parameters
    ----------
    labels : numpy.ndarray
        (N,) :class:`OrbitClass` integer values.
    circulation : numpy.ndarray
        (N, 3) ``|<L_a>| / <|L_a|>`` per axis.
    tube_axis : numpy.ndarray
        (N,) index (0=x, 1=y, 2=z) of the dominant circulation axis.
    planarity : numpy.ndarray
        (N,) ``lambda_min / lambda_max`` of the shape tensor.
    resonance : numpy.ndarray
        (N, 3) primitive resonance vector (all zero if none found).
    resonance_order : numpy.ndarray
        (N,) L1 order ``|n|_1`` of the resonance (0 if none found).
    class_names : dict, optional
        Mapping from integer label to name, used by :attr:`names` and
        :meth:`counts`. Defaults to the detailed :data:`CLASS_NAMES`; a condensed
        result (from :meth:`condense_families`) carries :data:`CONDENSED_NAMES`.
    radius : numpy.ndarray, optional
        (N,) characteristic radius of each orbit, in *physical* length units.
        This is the instantaneous snapshot radius (the orbit's radius at
        integration start, measured from the galaxy centre) and is the default
        binning radius for :meth:`plot_class_fractions`. Recorded by
        :func:`classify_orbits`; ``None`` when no radius is available.
    radius_orbit_averaged : numpy.ndarray, optional
        (N,) time-averaged radius ``r_mean`` of each orbit, in *physical* length
        units. Pass ``quantity="radius_orbit_averaged"`` to
        :meth:`plot_class_fractions` to bin on the orbit-averaged radius instead of the snapshot radius. Recorded by
        :func:`classify_orbits`; ``None`` when not available.
    ids : numpy.ndarray, optional
        (N,) particle ID of each orbit, recorded by :func:`classify_orbits` and
        used by :meth:`compare` to match particles between two classifications.
        ``None`` when not available.
    fundamentals : numpy.ndarray, optional
        (N, 3) signed fundamental frequency per axis, recorded by
        :func:`classify_orbits` when frequency data is available and used by
        :meth:`plot_frequency_map`. ``None`` when not available.
    quantities : dict, optional
        Extra per-orbit quantities to bin on in :meth:`fractions_by`, mapping a
        name to an (N,) array in *physical* units. :func:`classify_orbits`
        records ``"energy"`` (initial specific energy; the Jacobi integral for
        a rotating figure) and
        ``"angular_momentum"`` (``sqrt(sum_a <|L_a|>**2)``, the norm of the
        time-averaged absolute components -- a lower bound on the mean ``|L|``
        that, unlike ``|<L>|``, does not vanish for boxes). ``radius`` and
        ``radius_orbit_averaged`` are always available by those names.
    shape : numpy.ndarray, optional
        (N,) :class:`OrbitShape` values: the geometric shape of each orbit
        (box, tube type or rosette), assigned before the irregular override
        and without the boxlet split. Recorded by :func:`classify_orbits`;
        unaffected by :meth:`condense_families`.
    boxlet_type : numpy.ndarray, optional
        (N,) :class:`BoxletType` of each boxlet (``NONE`` otherwise), from its
        resonance vector. Like :attr:`shape` it is assigned before the
        irregular override, so a chaotic orbit that sits on a resonance keeps
        its type; combine with :attr:`labels` to select regular boxlets only.
    boxlet_plane : numpy.ndarray, optional
        (N,) axis perpendicular to the plane of a two-axis resonance (2 for an
        x-y boxlet); -1 for spatial boxlets and non-boxlets.
    boxlet_elongation : numpy.ndarray, optional
        (N,) axis the boxlet is elongated along (its lowest-frequency resonant
        axis); -1 for non-boxlets.
    boxlet_closed : numpy.ndarray, optional
        (N,) True for boxlets with a second, independent resonance
        (:attr:`second_resonance`): a closed (periodic) orbit rather than a
        thin one.
    second_resonance : numpy.ndarray, optional
        (N, 3) the second independent resonance vector of order >= 2 (all
        zero if none), for any orbit.
    sense : numpy.ndarray, optional
        (N,) :class:`OrbitSense` values (see :func:`classify_orbits`).
    orientation : str, optional
        The axis-sign reference of the classified orbits
        (:attr:`OrbitResults.orientation`), which sets what "prograde" means.
    pattern_resonance : numpy.ndarray, optional
        (N,) :class:`PatternResonance` values; all ``NONE`` for a static
        figure.
    pattern_frequency_ratio : numpy.ndarray, optional
        (N,) ``Omega_R / |Omega_phi,body|`` of the orbits tested for a pattern
        resonance (NaN otherwise), for checking the tolerance.
    regularity : numpy.ndarray, optional
        (N,) :class:`Regularity` values.
    unresolved : numpy.ndarray, optional
        (N,) True where a frequency one of the orbit's labels depends on
        completes fewer than ``resolution_cycles`` cycles in the window: an
        active axis fundamental (rosette and boxlet tests), the azimuthal
        frequency about a loop's axis (its sense), or the frequencies of the
        pattern-resonance test (``PatternResonance.UNRESOLVED``).
    pattern_speed_ratio : numpy.ndarray, optional
        (N,) each orbit's ``epsilon = |Omega_p| / |Omega_phi|``
        (:meth:`OrbitResults.pattern_speed_ratio`); 0 for a static figure, NaN
        for orbits not going round the pattern-speed axis.
    corotation_band : tuple of float, optional
        ``(low, high)`` range of ``epsilon`` flagged by
        :attr:`near_corotation`.
    inner_outer_ratio : float, optional
        The cut on ``|w_x| / sqrt(|w_y| |w_z|)`` used to split inner from outer
        long-axis tubes (see :func:`classify_orbits`).
    inner_outer_adaptive : bool, optional
        Whether that cut was placed adaptively from the population.
    family : numpy.ndarray, optional
        (N,) family index from the family table (see :func:`assign_families`).
    family_names : dict, optional
        Family index -> name for :attr:`family`.
    """

    labels: np.ndarray  # (N,) OrbitClass values
    circulation: np.ndarray  # (N,3) |<L_a>|/<|L_a|> per axis
    tube_axis: np.ndarray  # (N,) index of dominant circulation axis
    planarity: np.ndarray  # (N,) lambda_min/lambda_max of shape tensor
    resonance: np.ndarray  # (N,3) primitive resonance vector (0 if none)
    resonance_order: np.ndarray  # (N,) |n|_1 of the resonance (0 if none)
    class_names: Dict[int, str] = field(default_factory=lambda: dict(CLASS_NAMES))
    radius: Optional[np.ndarray] = None  # (N,) snapshot radius, physical units
    radius_orbit_averaged: Optional[np.ndarray] = None  # (N,) r_mean, physical
    ids: Optional[np.ndarray] = None  # (N,) particle IDs
    fundamentals: Optional[np.ndarray] = None  # (N,3) signed fund. freq per axis
    quantities: Optional[Dict[str, np.ndarray]] = None  # extra binning quantities
    shape: Optional[np.ndarray] = None  # (N,) OrbitShape values
    boxlet_type: Optional[np.ndarray] = None  # (N,) BoxletType values
    boxlet_plane: Optional[np.ndarray] = None  # (N,) normal axis of the plane, -1
    boxlet_elongation: Optional[np.ndarray] = None  # (N,) long axis, -1
    boxlet_closed: Optional[np.ndarray] = None  # (N,) two independent resonances
    second_resonance: Optional[np.ndarray] = None  # (N, 3) second vector
    sense: Optional[np.ndarray] = None  # (N,) OrbitSense values
    orientation: Optional[str] = None  # axis-sign reference of the results
    pattern_resonance: Optional[np.ndarray] = None  # (N,) PatternResonance
    pattern_frequency_ratio: Optional[np.ndarray] = None  # (N,) W_R/|W_phi,body|
    regularity: Optional[np.ndarray] = None  # (N,) Regularity values
    unresolved: Optional[np.ndarray] = None  # (N,) a label rests on a slow frequency
    pattern_speed_ratio: Optional[np.ndarray] = None  # (N,) |Omega_p| / |Omega_phi|
    corotation_band: Tuple[float, float] = DEFAULT_COROTATION_BAND
    inner_outer_ratio: Optional[float] = None  # inner/outer long-axis-tube cut
    inner_outer_adaptive: bool = False  # whether it was placed adaptively
    family: Optional[np.ndarray] = None  # (N,) family-table index
    family_names: Optional[Dict[int, str]] = None  # index -> family name

    # Label arrays and name tables selectable with ``scheme`` (see view()).
    _SCHEMES = {
        "family": ("family", "family_names"),
        "shape": ("shape", SHAPE_NAMES),
        "sense": ("sense", SENSE_NAMES),
        "pattern_resonance": ("pattern_resonance", PATTERN_RESONANCE_NAMES),
        "regularity": ("regularity", REGULARITY_NAMES),
        "boxlet": ("boxlet_type", BOXLET_TYPE_NAMES),
    }

    def view(self, scheme: str = "class") -> "OrbitClassification":
        """This classification with another label scheme as its ``labels``.

        Every method that works on :attr:`labels` / :attr:`class_names`
        (:meth:`counts`, :meth:`mask`, :meth:`fractions_by`, the plots,
        :meth:`compare`, ...) then works on that scheme.

        Parameters
        ----------
        scheme : str, optional
            ``"class"`` (the :class:`OrbitClass` labels; returns ``self``),
            ``"family"``, ``"shape"``, ``"sense"``, ``"pattern_resonance"``,
            ``"regularity"`` or ``"boxlet"``.

        Returns
        -------
        view : OrbitClassification
            A copy whose ``labels``/``class_names`` are those of ``scheme``.

        Raises
        ------
        ValueError
            If ``scheme`` is unknown or was not recorded.
        """
        if scheme == "class":
            return self
        if scheme not in self._SCHEMES:
            raise ValueError(
                f"scheme must be 'class' or one of {sorted(self._SCHEMES)}, got {scheme!r}"
            )
        array_name, names = self._SCHEMES[scheme]
        values = getattr(self, array_name)
        if isinstance(names, str):
            names = getattr(self, names)
        if values is None or names is None:
            raise ValueError(f"This classification has no {scheme!r} labels")
        return replace(self, labels=np.asarray(values).copy(), class_names=dict(names))

    @property
    def family_name_array(self) -> np.ndarray:
        """Per-orbit family names (see :attr:`family`).

        Returns
        -------
        names : numpy.ndarray
            (N,) strings.
        """
        return self.view("family").names

    def inconsistencies(self) -> Dict[str, int]:
        """Count orbits with label combinations that should never occur.

        Returns
        -------
        counts : dict
            Description -> number of orbits, for every check in
            :data:`INCONSISTENCY_CHECKS` (all zero for a sound classification).
        """
        return {name: int(np.sum(check(self))) for name, check in INCONSISTENCY_CHECKS}

    @property
    def regularity_names(self) -> np.ndarray:
        """Per-orbit regularity names (see :attr:`regularity`).

        Returns
        -------
        names : numpy.ndarray
            (N,) names from :data:`REGULARITY_NAMES`.

        Raises
        ------
        ValueError
            If no regularity was recorded.
        """
        if self.regularity is None:
            raise ValueError("This classification has no regularity labels")
        return np.array([REGULARITY_NAMES[int(r)] for r in self.regularity])

    @property
    def near_corotation(self) -> np.ndarray:
        """Orbits whose pattern speed ratio lies in :attr:`corotation_band`.

        Near and beyond corotation (by default ``epsilon >= 0.5``) the figure's
        rotation is a strong perturbation and the classification is least
        reliable. Orbits without a ratio
        (static figure, or not going round the pattern-speed axis) are not
        flagged.

        Returns
        -------
        mask : numpy.ndarray
            (N,) True where ``low <= epsilon <= high``.
        """
        if self.pattern_speed_ratio is None:
            return np.zeros(len(self.labels), dtype=bool)
        low, high = self.corotation_band
        with np.errstate(invalid="ignore"):
            return (self.pattern_speed_ratio >= low) & (
                self.pattern_speed_ratio <= high
            )

    @property
    def pattern_resonance_names(self) -> np.ndarray:
        """Per-orbit pattern-resonance names (see :attr:`pattern_resonance`).

        Returns
        -------
        names : numpy.ndarray
            (N,) names from :data:`PATTERN_RESONANCE_NAMES`.

        Raises
        ------
        ValueError
            If no pattern resonances were recorded.
        """
        if self.pattern_resonance is None:
            raise ValueError("This classification has no pattern resonances")
        return np.array(
            [PATTERN_RESONANCE_NAMES[int(r)] for r in self.pattern_resonance]
        )

    @property
    def corotation_trapped(self) -> np.ndarray:
        """Orbits librating at corotation (trapped around L4/L5).

        Returns
        -------
        mask : numpy.ndarray
            (N,) True where the sense is ``LIBRATING`` and the pattern
            resonance ``COROTATION``.

        Raises
        ------
        ValueError
            If the sense or pattern resonances were not recorded.
        """
        if self.sense is None or self.pattern_resonance is None:
            raise ValueError("This classification has no sense / pattern resonances")
        return (self.sense == OrbitSense.LIBRATING) & (
            self.pattern_resonance == PatternResonance.COROTATION
        )

    @property
    def sense_names(self) -> np.ndarray:
        """Per-orbit sense name strings (see :attr:`sense`).

        Returns
        -------
        names : numpy.ndarray
            (N,) names from :data:`SENSE_NAMES`.

        Raises
        ------
        ValueError
            If no sense was recorded.
        """
        if self.sense is None:
            raise ValueError("This classification has no sense labels")
        return np.array([SENSE_NAMES[int(s)] for s in self.sense])

    @property
    def shape_names(self) -> np.ndarray:
        """Per-orbit shape name strings (see :attr:`shape`).

        Returns
        -------
        names : numpy.ndarray
            (N,) names from :data:`SHAPE_NAMES`.

        Raises
        ------
        ValueError
            If no shape was recorded.
        """
        if self.shape is None:
            raise ValueError("This classification has no shape labels")
        return np.array([SHAPE_NAMES[int(s)] for s in self.shape])

    @property
    def boxlet_names(self) -> np.ndarray:
        """Per-orbit boxlet name (empty string for non-boxlets).

        A named two-axis boxlet is ``"<type>_<plane>"`` (e.g. ``"banana_xy"``),
        another two-axis one ``"<p>:<q>_<plane>"`` (lower:higher in-plane
        frequency ratio, e.g. ``"1:3_xz"``), and a spatial one
        ``"resonant_<nx>_<ny>_<nz>"``.

        Returns
        -------
        names : numpy.ndarray
            (N,) strings.

        Raises
        ------
        ValueError
            If no boxlet types were recorded.
        """
        if self.boxlet_type is None:
            raise ValueError("This classification has no boxlet types")
        names = np.full(len(self.boxlet_type), "", dtype=object)
        for k in np.flatnonzero(self.boxlet_type != BoxletType.NONE):
            n = [int(c) for c in self.resonance[k]]
            kind = BoxletType(int(self.boxlet_type[k]))
            if kind == BoxletType.SPATIAL:
                names[k] = f"resonant_{n[0]}_{n[1]}_{n[2]}"
                continue
            plane = "".join("xyz"[a] for a in range(3) if a != self.boxlet_plane[k])
            if kind == BoxletType.PLANAR:
                p, q = sorted(abs(c) for c in n if c != 0)
                names[k] = f"{p}:{q}_{plane}"
            else:
                names[k] = f"{BOXLET_TYPE_NAMES[int(kind)]}_{plane}"
        return names.astype(str)

    def boxlet_counts(self, detailed: bool = False, particles=None) -> dict:
        """Count the boxlets (or their mass) of each type.

        Parameters
        ----------
        detailed : bool, optional
            If True, count by :attr:`boxlet_names` (type, plane and resonance)
            instead of by :class:`BoxletType`.
        particles : lanfear.ParticleSystem, optional
            If given, sum the particle masses matched by ID instead of counting.

        Returns
        -------
        counts : dict
            Name -> number of boxlets (or total mass), largest first.
        """
        boxlet = self.boxlet_type != BoxletType.NONE
        if detailed:
            names = self.boxlet_names[boxlet]
        else:
            names = np.array(
                [BOXLET_TYPE_NAMES[int(t)] for t in self.boxlet_type[boxlet]]
            )
        if not len(names):
            return {}
        values, inverse = np.unique(names, return_inverse=True)
        if particles is None:
            counts = np.bincount(inverse.ravel(), minlength=len(values))
            cast = int
        else:
            weights = particle_weights(self.ids, particles)[boxlet]
            counts = np.bincount(
                inverse.ravel(), weights=weights, minlength=len(values)
            )
            cast = float
        order = np.argsort(-counts, kind="stable")
        return {str(values[i]): cast(counts[i]) for i in order}

    @property
    def names(self) -> np.ndarray:
        """Per-orbit class name strings.

        Returns
        -------
        names : numpy.ndarray
            (N,) lower-case family names (e.g. ``"short_axis_tube"``, or
            ``"tube"``/``"box"`` for a condensed result).
        """
        return np.array([self.class_names[int(v)] for v in self.labels])

    def counts(self, scheme: str = "class", particles=None) -> dict:
        """Count the orbits (or their mass) in each family.

        Parameters
        ----------
        scheme : str, optional
            Label scheme to count (see :meth:`view`; default the class labels).
        particles : lanfear.ParticleSystem, optional
            If given, sum the masses of the particles matched by ID
            (:func:`particle_weights`) instead of counting orbits.

        Returns
        -------
        counts : dict
            Mapping from class name to the number of orbits in that family
            (int), or their total mass (float) when ``particles`` is given.
        """
        if scheme != "class":
            return self.view(scheme).counts(particles=particles)
        if particles is not None:
            weights = particle_weights(self.ids, particles)
            vals, inverse = np.unique(self.labels, return_inverse=True)
            mass = np.bincount(inverse.ravel(), weights=weights, minlength=len(vals))
            return {self.class_names[int(v)]: float(m) for v, m in zip(vals, mass)}
        vals, cnts = np.unique(self.labels, return_counts=True)
        return {self.class_names[int(v)]: int(c) for v, c in zip(vals, cnts)}

    def mask(self, cls, scheme: str = "class") -> np.ndarray:
        """Boolean mask selecting orbits of a given family or boxlet type.

        Parameters
        ----------
        cls : OrbitClass, OrbitFamily, BoxletType, int or str
            The family to select (e.g. ``OrbitClass.SHORT_AXIS_TUBE`` on a full
            result, or ``OrbitFamily.TUBE`` on a condensed one), a boxlet type
            (``BoxletType.BANANA``), or a name: a family name (``"boxlet"``),
            a boxlet type name (``"banana"``) or a detailed boxlet name
            (``"banana_xy"``, see :attr:`boxlet_names`).
        scheme : str, optional
            Label scheme ``cls`` refers to (see :meth:`view`), e.g.
            ``mask("x1", scheme="family")``. Default: the class labels (with
            the boxlet shortcuts above).

        Returns
        -------
        mask : numpy.ndarray
            (N,) True where the orbit belongs to ``cls``.

        Raises
        ------
        ValueError
            If ``cls`` is a name that matches nothing.
        """
        if scheme != "class":
            view = self.view(scheme)
            if isinstance(cls, str):
                by_name = {name: value for value, name in view.class_names.items()}
                if cls not in by_name:
                    raise ValueError(f"{cls!r} is not a {scheme!r} name")
                return view.labels == by_name[cls]
            return view.labels == int(cls)
        if isinstance(cls, BoxletType):
            return self.boxlet_type == int(cls)
        if isinstance(cls, str):
            by_name = {name: value for value, name in self.class_names.items()}
            if cls in by_name:
                return self.labels == by_name[cls]
            types = {name: value for value, name in BOXLET_TYPE_NAMES.items()}
            if cls in types and cls != "none" and self.boxlet_type is not None:
                return self.boxlet_type == types[cls]
            if self.boxlet_type is not None:
                names = self.boxlet_names
                if np.any(names == cls):
                    return names == cls
            raise ValueError(f"{cls!r} is not a family, boxlet type or boxlet name")
        return self.labels == int(cls)

    def get_class_ids(self, cls, scheme: str = "class") -> np.ndarray:
        """Particle IDs of the orbits belonging to a given family.

        Parameters
        ----------
        cls : OrbitClass, OrbitFamily, BoxletType, int or str
            The family or boxlet type to select (see :meth:`mask`).
        scheme : str, optional
            Label scheme ``cls`` refers to (see :meth:`mask`).

        Returns
        -------
        ids : numpy.ndarray
            Particle IDs of the orbits belonging to ``cls``.

        Raises
        ------
        ValueError
            If this classification carries no particle IDs.
        """
        if self.ids is None:
            raise ValueError(
                "no particle IDs available; build the classification with "
                "classify_orbits (which records them from OrbitResults.ids)."
            )
        return self.ids[self.mask(cls, scheme)]

    def condense_families(self) -> "OrbitClassification":
        """Group the orbit subclasses into the box/tube dichotomy.

        All tube subclasses (short-axis, inner/outer long-axis and
        intermediate-axis tubes, and rosettes) fold into
        :attr:`OrbitFamily.TUBE`; boxes and boxlets fold into
        :attr:`OrbitFamily.BOX`; unclassified orbits stay unclassified. The
        per-orbit diagnostic arrays (circulation, tube axis, planarity,
        resonance) are carried through unchanged.

        Returns
        -------
        condensed : OrbitClassification
            A new classification whose ``labels`` are :class:`OrbitFamily`
            values and whose :attr:`names`/:meth:`counts` report ``"box"``,
            ``"tube"`` or ``"unclassified"``.
        """
        if self.class_names not in (CLASS_NAMES, CONDENSED_NAMES):
            raise ValueError(
                "condense_families() works on the class labels, not on a view of "
                "another scheme"
            )
        if self.class_names == CONDENSED_NAMES:
            # Already condensed -- return an equivalent copy (idempotent). The
            # OrbitClass -> family map cannot be reapplied to family labels.
            return replace(
                self, labels=self.labels.copy(), class_names=dict(CONDENSED_NAMES)
            )
        condensed = np.full(self.labels.shape, OrbitFamily.UNCLASSIFIED, dtype=np.int64)
        for cls in _TUBE_CLASSES:
            condensed[self.labels == int(cls)] = OrbitFamily.TUBE
        for cls in _BOX_CLASSES:
            condensed[self.labels == int(cls)] = OrbitFamily.BOX
        return replace(self, labels=condensed, class_names=dict(CONDENSED_NAMES))

    def _binning_values(self, quantity) -> tuple:
        """Resolve ``quantity`` to an (N,) array and an axis label.

        Parameters
        ----------
        quantity : str or array_like
            ``"radius"``, ``"radius_orbit_averaged"``, a key of
            :attr:`quantities`, or an (N,) array of your own.

        Returns
        -------
        values : numpy.ndarray
            (N,) float array.
        label : str
            Axis label for the quantity.

        Raises
        ------
        ValueError
            If the quantity is unknown, unavailable, or the wrong length.
        """
        labels = {
            "radius": "radius (physical units)",
            "radius_orbit_averaged": "orbit-averaged radius (physical units)",
            "energy": "specific energy (physical units)",
            "angular_momentum": "angular momentum (physical units)",
            "pattern_speed_ratio": r"$|\Omega_\mathrm{p}| / |\Omega_\phi|$",
        }
        if isinstance(quantity, str):
            available = {
                "radius": self.radius,
                "radius_orbit_averaged": self.radius_orbit_averaged,
            }
            available.update(self.quantities or {})
            if quantity not in available:
                raise ValueError(
                    f"unknown quantity {quantity!r}; choose from "
                    f"{sorted(available)} or pass an (N,) array."
                )
            values = available[quantity]
            if values is None:
                raise ValueError(
                    f"quantity {quantity!r} is not available; build the "
                    "classification with classify_orbits or pass an array."
                )
            label = labels.get(quantity, quantity)
        else:
            values, label = quantity, "binned quantity"
        values = np.asarray(values, dtype=float)
        if values.shape != self.labels.shape:
            raise ValueError("quantity and labels must have the same length.")
        return values, label

    def fractions_by(
        self,
        edges,
        quantity="radius",
        per_bin: bool = True,
        n_bootstrap: int = 0,
        confidence: float = 0.68,
        seed=None,
        scheme: str = "class",
        particles=None,
    ) -> "ClassFractions":
        """Fraction of orbits in each class within bins of a per-orbit quantity.

        Parameters
        ----------
        edges : array_like
            (n_bins + 1,) monotonically increasing bin edges, in the units of
            ``quantity``. Orbits outside the edges are ignored.
        quantity : str or array_like, optional
            What to bin on: ``"radius"`` (snapshot radius, default),
            ``"radius_orbit_averaged"``, ``"energy"``, ``"angular_momentum"``
            (see :attr:`quantities`), or an (N,) array of your own.
        per_bin : bool, optional
            If ``True`` (default), fractions are the class composition within
            each bin; if ``False``, each class count is divided by the total
            number of binned orbits.
        n_bootstrap : int, optional
            Number of bootstrap replicates used to estimate the uncertainty.
            ``0`` (default) skips it. Each replicate resamples the orbits with
            replacement; this is done exactly and cheaply by drawing the cell
            counts from a multinomial, so it is fast even for millions of
            orbits. It captures sampling noise from the finite number of orbits
            only, not systematic effects (e.g. integration length), and the
            interval collapses to zero width where a class is absent.
        confidence : float, optional
            Central confidence level of the interval (default 0.68, ~1 sigma).
        seed : int or numpy.random.Generator, optional
            Seed for the bootstrap.
        scheme : str, optional
            Label scheme to compute fractions of (see :meth:`view`).
        particles : lanfear.ParticleSystem, optional
            If given, weight each orbit by the mass of the particle with its ID
            (:func:`particle_weights`): the fractions become mass fractions,
            and the bootstrap resamples orbits, carrying their masses.

        Returns
        -------
        fractions : ClassFractions
            Fractions, counts and (if requested) bootstrap bounds.

        Raises
        ------
        ValueError
            If ``edges`` or ``quantity`` are malformed, or ``confidence`` is not
            in (0, 1).
        """
        if scheme != "class":
            return self.view(scheme).fractions_by(
                edges,
                quantity,
                per_bin,
                n_bootstrap,
                confidence,
                seed,
                particles=particles,
            )
        weighted = particles is not None
        weights = (
            particle_weights(self.ids, particles)
            if weighted
            else np.ones(len(self.labels))
        )
        values, label = self._binning_values(quantity)
        if isinstance(edges, int):
            edges = np.geomspace(np.nanmin(values), np.nanmax(values), edges)
        edges = np.asarray(edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("edges must be a 1-D array of at least two bin edges.")
        if np.any(np.diff(edges) <= 0):
            raise ValueError("edges must be strictly increasing.")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must lie in (0, 1).")
        n_bins = edges.size - 1

        classes = np.array(sorted(int(v) for v in np.unique(self.labels)), dtype=int)
        n_classes = classes.size
        class_index = np.searchsorted(classes, self.labels)
        bin_index = np.digitize(values, edges) - 1
        in_range = (bin_index >= 0) & (bin_index < n_bins)
        if not in_range.any():
            logger.warning("No orbits fall within the given edges.")

        counts = np.zeros((n_classes, n_bins))
        np.add.at(
            counts, (class_index[in_range], bin_index[in_range]), weights[in_range]
        )

        def _normalise(cell_counts):
            total = cell_counts.sum(axis=-2 if per_bin else (-2, -1), keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                # Empty bins have no defined composition -> NaN (a gap).
                return cell_counts / np.where(total == 0, np.nan, total)

        fractions = _normalise(counts)
        lower = upper = None
        if n_bootstrap > 0:
            rng = np.random.default_rng(seed)
            if not weighted:
                # Resampling N orbits with replacement is a multinomial over the
                # cells (plus one "out of range" cell), so no per-orbit work.
                cells = np.append(counts.ravel(), float((~in_range).sum()))
                draws = rng.multinomial(
                    self.labels.size, cells / cells.sum(), size=int(n_bootstrap)
                ).astype(float)
            else:
                # With unequal masses, resample the orbits themselves and sum
                # the masses they carry into each cell.
                n_cells = n_classes * n_bins
                cell = np.where(
                    in_range,
                    class_index * n_bins + np.clip(bin_index, 0, None),
                    n_cells,
                )
                draws = np.empty((int(n_bootstrap), n_cells + 1))
                for k in range(int(n_bootstrap)):
                    pick = rng.integers(0, len(cell), len(cell))
                    draws[k] = np.bincount(
                        cell[pick], weights=weights[pick], minlength=n_cells + 1
                    )
            replicas = _normalise(draws[:, :-1].reshape(-1, n_classes, n_bins))
            tail = 50.0 * (1.0 - confidence)
            with np.errstate(all="ignore"):
                lower = np.nanpercentile(replicas, tail, axis=0)
                upper = np.nanpercentile(replicas, 100.0 - tail, axis=0)

        return ClassFractions(
            edges=edges,
            centres=0.5 * (edges[:-1] + edges[1:]),
            labels=classes,
            class_names={int(c): self.class_names[int(c)] for c in classes},
            counts=counts,
            fractions=fractions,
            lower=lower,
            upper=upper,
            per_bin=per_bin,
            quantity_label=label,
            weighted=weighted,
        )

    def plot_class_fractions(
        self,
        edges,
        per_bin: bool = True,
        ax=None,
        quantity="radius",
        n_bootstrap: int = 0,
        confidence: float = 0.68,
        seed=None,
        scheme: str = "class",
        particles=None,
        **kwargs,
    ):
        """Plot the relative frequency of each orbit class in bins.

        Orbits are binned by a per-orbit quantity (the characteristic radius by
        default), and the fraction of orbits belonging to each class is drawn as
        a curve against it (one line per class present). See
        :meth:`fractions_by` for the numbers behind the plot.

        Parameters
        ----------
        edges : array_like
            (n_bins + 1,) monotonically increasing bin edges, in the units of
            the binned quantity (*physical* units for the built-in ones).
        per_bin : bool, optional
            Normalisation of the frequencies. If ``True`` (default), each
            class count in a bin is divided by the number of orbits in that
            bin, so the curves give the class composition within each bin
            (summing to 1 across classes). If ``False``, counts are divided by
            the total number of binned orbits, so the curves give each class's
            share of the whole population.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. A new figure and axes are created if omitted.
        quantity : str or array_like, optional
            What to bin on; see :meth:`fractions_by`. Default ``"radius"``.
        n_bootstrap : int, optional
            If > 0, shade the bootstrap confidence band of each curve.
        confidence : float, optional
            Confidence level of the band (default 0.68).
        seed : int or numpy.random.Generator, optional
            Seed for the bootstrap.
        scheme : str, optional
            Label scheme to plot (see :meth:`view`; default the class labels).
        particles : lanfear.ParticleSystem, optional
            If given, plot mass fractions, weighting each orbit by the mass of
            the particle with its ID (see :meth:`fractions_by`).
        **kwargs
            Passed through to ``ax.plot`` for every class. Pass ``color=...``
            to override :data:`DEFAULT_PALETTE` for all classes at once.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes the curves were drawn on.

        Raises
        ------
        ValueError
            If the quantity is unavailable, or ``quantity``/``edges`` are
            malformed.
        ImportError
            If matplotlib is not installed.
        """
        result = self.fractions_by(
            edges,
            quantity=quantity,
            per_bin=per_bin,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed,
            scheme=scheme,
            particles=particles,
        )
        return result.plot(ax=ax, **kwargs)

    def plot_class_histograms(
        self, ax=None, scheme: str = "class", particles=None, **kwargs
    ):
        """Bar chart of the number of orbits in each class.

        Draws one bar per populated class, with the class name as a categorical
        x-axis label and the bar height the number of orbits (from
        :meth:`counts`).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. A new figure and axes are created if omitted.
        scheme : str, optional
            Label scheme to count (see :meth:`view`; default the class labels).
        particles : lanfear.ParticleSystem, optional
            If given, bar heights are the total mass of each class (particles
            matched by ID) instead of orbit counts.
        **kwargs
            Passed through to ``ax.bar``. Pass ``color=...`` to override
            :data:`DEFAULT_PALETTE`.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes the bars were drawn on.
        """
        counts = self.counts(scheme, particles=particles)
        names = list(counts.keys())
        values = [counts[name] for name in names]

        if ax is None:
            _, ax = plt.subplots()

        positions = np.arange(len(names))
        bar_kwargs = dict(kwargs)
        bar_kwargs.setdefault(
            "color", [_colour_for(name, i) for i, name in enumerate(names)]
        )
        bar_kwargs.setdefault("lw", 0.3)
        bar_kwargs.setdefault("ec", "k")
        ax.bar(positions, values, **bar_kwargs)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [_latex_label(name) for name in names], rotation=45, ha="right"
        )
        ax.set_xlabel("orbit class")
        ax.set_ylabel("number of orbits" if particles is None else "mass")
        # Reserve room for the rotated tick labels so they are not clipped when
        # the figure is saved with a plain savefig() (no bbox_inches="tight").
        ax.figure.tight_layout()
        return ax

    def plot_frequency_map(
        self,
        ax=None,
        colourmap: Optional[str] = None,
        marker_size: float = 6.0,
        alpha: float = 0.8,
        legend: bool = True,
        by: str = "class",
        resonance_lines: bool = False,
        max_lines: int = 8,
    ):
        """Frequency map: scatter of the fundamental-frequency ratios by class.

        Each orbit is plotted at ``(|w_x| / |w_z|, |w_y| / |w_z|)`` and coloured
        by its orbit class. In such a map regular families cluster and low-order
        resonances trace straight lines, so it is a compact visual summary of the
        orbital structure. Requires frequency data (the classification must come
        from :func:`analyse_family` / :func:`analyse_states`).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. A new figure and axes are created if omitted.
        colourmap : str, optional
            Name of a matplotlib colormap; each class takes the colour at its
            integer label, so colours are stable across plots. Defaults to
            :data:`DEFAULT_PALETTE` when omitted.
        marker_size : float, optional
            Scatter marker size (points**2).
        alpha : float, optional
            Marker opacity (helps with dense, overlapping points).
        legend : bool, optional
            Whether to draw the class legend.
        by : str, optional
            Colour by orbit class (default), by :class:`BoxletType` with all
            non-boxlets in one grey group (``"boxlet"``), or by any other label
            scheme (``"family"``, ``"shape"``, ``"sense"``, ...; see
            :meth:`view`).
        resonance_lines : bool, optional
            Overlay the resonance lines ``n_x X + n_y Y + n_z = 0`` (with
            ``X = |w_x|/|w_z|``, ``Y = |w_y|/|w_z|``) of the most common boxlet
            resonance vectors.
        max_lines : int, optional
            Maximum number of resonance lines drawn (most common first).

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes the points were drawn on.

        Raises
        ------
        ValueError
            If the classification carries no frequency data, or ``by`` is not
            recognised.
        """
        if self.fundamentals is None:
            raise ValueError(
                "no frequency data; classify results from analyse_family/"
                "analyse_states to populate the fundamental frequencies."
            )
        w = np.abs(np.asarray(self.fundamentals, dtype=np.float64))
        wz = w[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_x = w[:, 0] / wz
            ratio_y = w[:, 1] / wz
        finite = (wz > 0) & np.isfinite(ratio_x) & np.isfinite(ratio_y)

        if by == "class":
            groups, group_names = self.labels, self.class_names
        elif by == "boxlet":
            groups, group_names = self.boxlet_type, dict(BOXLET_TYPE_NAMES)
            group_names[int(BoxletType.NONE)] = "not a boxlet"
        else:
            view = self.view(by)
            groups, group_names = view.labels, view.class_names

        if ax is None:
            _, ax = plt.subplots()
        cmap = plt.get_cmap(colourmap) if colourmap is not None else None
        for i, cls in enumerate(sorted(int(v) for v in np.unique(groups))):
            sel = finite & (groups == cls)
            if not np.any(sel):
                continue
            if by == "boxlet" and cls == BoxletType.NONE:
                colour = "0.8"
            elif cmap is not None:
                colour = cmap(cls % cmap.N)
            else:
                colour = _colour_for(group_names[cls], i)
            ax.scatter(
                ratio_x[sel],
                ratio_y[sel],
                s=marker_size,
                alpha=alpha,
                color=colour,
                edgecolors="none",
                label=_latex_label(group_names[cls]),
            )
        if resonance_lines:
            self._draw_resonance_lines(ax, max_lines)
        ax.set_xlabel(r"$|\omega_x| / |\omega_z|$")
        ax.set_ylabel(r"$|\omega_y| / |\omega_z|$")
        if legend:
            ax.legend(
                title={"class": "orbit class", "boxlet": "boxlet type"}.get(by, by),
                markerscale=2.0,
                framealpha=0.9,
                loc="upper left",
            )
        ax.figure.tight_layout()
        return ax

    def _draw_resonance_lines(self, ax, max_lines: int) -> None:
        """Draw the most common boxlet resonances on a frequency map.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Frequency-map axes (``X = |w_x|/|w_z|``, ``Y = |w_y|/|w_z|``).
        max_lines : int
            Maximum number of lines.
        """
        if self.boxlet_type is None:
            raise ValueError("This classification has no boxlet types")
        vectors = self.resonance[self.boxlet_type != BoxletType.NONE]
        if not len(vectors):
            return
        unique, counts = np.unique(vectors, axis=0, return_counts=True)
        x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
        x = np.linspace(*x_lim, 200)
        for n in unique[np.argsort(-counts, kind="stable")][:max_lines]:
            nx, ny, nz = (int(c) for c in n)
            if ny != 0:
                ax.plot(x, -(nx * x + nz) / ny, color="k", lw=0.6, ls="--", alpha=0.6)
            elif nx != 0:
                ax.axvline(-nz / nx, color="k", lw=0.6, ls="--", alpha=0.6)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    def compare(
        self,
        other: Union["OrbitClassification", Iterable["OrbitClassification"]],
        scheme: str = "class",
    ) -> "ClassificationComparison":
        """Compare per-particle classifications across two or more snapshots.

        Particles are matched by their ID (:attr:`ids`), so only those present
        in *every* classification are compared; particles missing from any one
        snapshot are dropped. This is useful for tracking how orbit families
        change as a system evolves, either between two snapshots or across a
        whole sequence of them (e.g. to draw a multi-column
        :meth:`ClassificationComparison.plot_sankey`).

        No temporal order is assumed: ``self`` is treated as the first stage
        and ``other`` (or each element of ``other``, in order) as the
        subsequent stages, purely by convention -- it is left to the caller to
        pass them in the order they should be compared.

        Parameters
        ----------
        other : OrbitClassification or iterable of OrbitClassification
            The classification(s) to compare against. A single
            :class:`OrbitClassification` compares two stages (as before); an
            iterable of them compares ``self`` followed by each item, in
            order, producing one stage per classification.
        scheme : str, optional
            Label scheme to compare (see :meth:`view`), applied to every
            classification; default the class labels.

        Returns
        -------
        comparison : ClassificationComparison
            Per-particle labels/names at each stage for the matched
            particles, a mask of which particles changed family at any point
            in the sequence, and the family transitions between consecutive
            stages.

        Raises
        ------
        ValueError
            If ``other`` is an empty iterable, if any classification carries
            no particle IDs, if the classifications use different class
            schemes (e.g. comparing a full classification against a
            condensed one), or if their axis signs were fixed differently
            (:attr:`orientation`), so that prograde and retrograde would not
            mean the same.
        """
        if scheme != "class":
            if isinstance(other, OrbitClassification):
                other = other.view(scheme)
            else:
                other = [state.view(scheme) for state in other]
            return self.view(scheme).compare(other)
        if isinstance(other, OrbitClassification):
            others = [other]
        else:
            others = list(other)
            if not others:
                raise ValueError(
                    "other must be an OrbitClassification or a non-empty "
                    "iterable of them."
                )
        states = [self] + others

        for state in states:
            if state.ids is None:
                raise ValueError(
                    "all classifications must carry particle IDs to be "
                    "compared; build them with classify_orbits (which "
                    "records ids)."
                )
        for state in others:
            if state.orientation != self.orientation:
                raise ValueError(
                    f"cannot compare classifications whose axis signs were fixed "
                    f"differently (orientation {self.orientation!r} vs "
                    f"{state.orientation!r}); orient both with the same "
                    f"reference (ParticleSystem.orient_axes)"
                )
            if state.class_names != self.class_names:
                raise ValueError(
                    "cannot compare classifications with different class "
                    "schemes (e.g. a full classification against a condensed "
                    "one); condense all of them with condense_families() "
                    "first, or compare classifications that all use the "
                    "same scheme."
                )

        ids_arrays = [np.asarray(state.ids) for state in states]
        common = ids_arrays[0]
        for ids in ids_arrays[1:]:
            common = np.intersect1d(common, ids)

        idx_list = []
        for ids in ids_arrays:
            order = np.argsort(ids)
            idx_list.append(order[np.searchsorted(ids[order], common)])

        sizes = "/".join(str(len(ids)) for ids in ids_arrays)
        logger.info(
            f"Comparing {len(states)} classifications: {len(common)} matched "
            f"of {sizes} particles"
        )

        labels = [np.asarray(state.labels)[idx] for state, idx in zip(states, idx_list)]
        names = [
            np.array([state.class_names[int(v)] for v in lab])
            for state, lab in zip(states, labels)
        ]
        changed = np.zeros(len(common), dtype=bool)
        for before, after in zip(names[:-1], names[1:]):
            changed |= before != after

        return ClassificationComparison(
            ids=common,
            labels=labels,
            names=names,
            changed=changed,
            class_names=[dict(state.class_names) for state in states],
        )


@dataclass
class ClassificationComparison:
    """Per-particle comparison of two or more :class:`OrbitClassification` stages.

    Produced by :meth:`OrbitClassification.compare`. All per-particle arrays
    are aligned and indexed identically, ordered by matched particle ID.
    ``labels``/``names``/``class_names`` hold one entry per stage, in the
    order passed to :meth:`OrbitClassification.compare` (``self`` first, then
    each ``other``). ``before``/``after`` refer to the first and last stage
    respectively, by convention only, and remain available as a convenience
    for the common two-stage comparison.

    Parameters
    ----------
    ids : numpy.ndarray
        (M,) particle IDs present in every stage, sorted ascending.
    labels : list of numpy.ndarray
        One (M,) array of integer class labels per stage.
    names : list of numpy.ndarray
        One (M,) array of class-name strings per stage, corresponding to
        ``labels``.
    changed : numpy.ndarray
        (M,) boolean, ``True`` where a particle's family name differs between
        any two consecutive stages.
    class_names : list of dict
        One label-to-name mapping per stage (used to build the transition
        matrices).
    """

    ids: np.ndarray
    labels: List[np.ndarray]
    names: List[np.ndarray]
    changed: np.ndarray
    class_names: List[Dict[int, str]]

    @property
    def n_matched(self) -> int:
        """Number of particles present in every stage.

        Returns
        -------
        n_matched : int
            Count of matched particles.
        """
        return int(len(self.ids))

    @property
    def n_stages(self) -> int:
        """Number of classification stages being compared.

        Returns
        -------
        n_stages : int
            ``2`` for a plain before/after comparison, more for a comparison
            built from an iterable of classifications.
        """
        return len(self.labels)

    @property
    def fraction_changed(self) -> float:
        """Fraction of matched particles that changed family at any point.

        Returns
        -------
        fraction_changed : float
            ``changed.sum() / n_matched`` (0.0 if no particles matched).
        """
        if self.n_matched == 0:
            return 0.0
        return float(np.sum(self.changed) / self.n_matched)

    @property
    def labels_before(self) -> np.ndarray:
        """Integer class labels at the first stage (see :attr:`labels`)."""
        return self.labels[0]

    @property
    def labels_after(self) -> np.ndarray:
        """Integer class labels at the last stage (see :attr:`labels`)."""
        return self.labels[-1]

    @property
    def names_before(self) -> np.ndarray:
        """Class names at the first stage (see :attr:`names`)."""
        return self.names[0]

    @property
    def names_after(self) -> np.ndarray:
        """Class names at the last stage (see :attr:`names`)."""
        return self.names[-1]

    @property
    def class_names_before(self) -> Dict[int, str]:
        """Label-to-name mapping at the first stage (see :attr:`class_names`)."""
        return self.class_names[0]

    @property
    def class_names_after(self) -> Dict[int, str]:
        """Label-to-name mapping at the last stage (see :attr:`class_names`)."""
        return self.class_names[-1]

    def transition_matrix(self, stage: int = 0, particles=None):
        """Count matrix of family transitions between two consecutive stages.

        Parameters
        ----------
        stage : int, optional
            Index of the earlier stage in the pair to compare (``0`` is
            ``self`` vs the first ``other`` passed to
            :meth:`OrbitClassification.compare`). Defaults to ``0``, the only
            valid value for a plain two-stage (before/after) comparison.
        particles : lanfear.ParticleSystem, optional
            If given, sum the masses of the matched particles (by ID, see
            :func:`particle_weights`) instead of counting them.

        Returns
        -------
        row_names : list of str
            Class names of stage ``stage``, one per matrix row.
        col_names : list of str
            Class names of stage ``stage + 1``, one per matrix column.
        matrix : numpy.ndarray
            ``(len(row_names), len(col_names))`` integer counts (float masses
            when ``particles`` is given), where ``matrix[i, j]`` is the number
            of particles classified as ``row_names[i]`` at stage ``stage`` and
            ``col_names[j]`` at stage ``stage + 1``.

        Raises
        ------
        ValueError
            If ``stage`` does not index a valid consecutive stage pair.
        """
        if not 0 <= stage < self.n_stages - 1:
            raise ValueError(
                f"stage must be in [0, {self.n_stages - 2}] for a comparison "
                f"with {self.n_stages} stages."
            )
        before_all = self.labels[stage]
        after_all = self.labels[stage + 1]
        before_names = self.class_names[stage]
        after_names = self.class_names[stage + 1]

        before_labels = sorted({int(v) for v in before_all})
        after_labels = sorted({int(v) for v in after_all})
        row_of = {lab: i for i, lab in enumerate(before_labels)}
        col_of = {lab: j for j, lab in enumerate(after_labels)}
        if particles is None:
            weights = np.ones(len(before_all), dtype=np.int64)
            matrix = np.zeros((len(before_labels), len(after_labels)), dtype=np.int64)
        else:
            weights = particle_weights(self.ids, particles)
            matrix = np.zeros((len(before_labels), len(after_labels)))
        for before, after, weight in zip(before_all, after_all, weights):
            matrix[row_of[int(before)], col_of[int(after)]] += weight
        row_names = [before_names[lab] for lab in before_labels]
        col_names = [after_names[lab] for lab in after_labels]
        return row_names, col_names, matrix

    def plot_sankey(
        self,
        ax=None,
        colourmap: Optional[str] = None,
        node_width: float = 0.03,
        alpha: float = 0.6,
        stage_labels: Optional[list] = None,
        particles=None,
    ):
        """Draw a Sankey diagram of the family flow across two or more stages.

        Draws one node column per classification stage (``self`` followed, in
        order, by each ``other`` passed to
        :meth:`OrbitClassification.compare`), with ribbons between every pair
        of consecutive columns whose width is proportional to the number of
        particles that moved from the source family to the destination
        family. Node heights are the family totals within that stage, so a
        family that keeps most of its members shows one dominant self-flow.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. A new figure and axes are created if omitted.
        colourmap : str, optional
            Name of a matplotlib colormap used to colour the families; each
            ribbon takes the colour of its source family. Defaults to
            :data:`DEFAULT_PALETTE` when omitted.
        node_width : float, optional
            Width of the node bars as a fraction of the horizontal span.
        alpha : float, optional
            Opacity of the flow ribbons.
        stage_labels : sequence of str, optional
            One label per stage, drawn above its node column. Defaults to
            ``["this", "other"]`` for a two-stage comparison, or
            ``["stage 0", "stage 1", ...]`` for more stages.
        particles : lanfear.ParticleSystem, optional
            If given, ribbon widths and node totals are particle masses
            (matched by ID) rather than particle counts.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes the diagram was drawn on.

        Raises
        ------
        ValueError
            If ``stage_labels`` is given and does not have one entry per
            stage.
        """
        from matplotlib.patches import PathPatch, Rectangle
        from matplotlib.path import Path

        n_stages = self.n_stages
        if stage_labels is None:
            stage_labels = (
                ["this", "other"]
                if n_stages == 2
                else [f"stage {k}" for k in range(n_stages)]
            )
        elif len(stage_labels) != n_stages:
            raise ValueError(
                f"stage_labels must have {n_stages} entries (one per stage)."
            )

        transitions = [
            self.transition_matrix(k, particles=particles) for k in range(n_stages - 1)
        ]
        total = float(transitions[0][2].sum())

        if ax is None:
            _, ax = plt.subplots()
        if total == 0:
            logger.warning("No matched particles; nothing to draw.")
            ax.axis("off")
            return ax
        gap = 0.02 * total  # vertical space between stacked nodes

        def _stack(sizes):
            # Top y of each node, stacked downward and centred about y = 0.
            column_height = sizes.sum() + gap * (len(sizes) - 1)
            tops = np.empty(len(sizes))
            y = 0.5 * column_height
            for i, h in enumerate(sizes):
                tops[i] = y
                y -= h + gap
            return tops

        # Names/sizes/tops for every stage. A stage's composition is read off
        # the row side of its outgoing transition (or the column side of its
        # incoming one for the last stage) -- both describe the same matched
        # population, since every transition is built from the same ids.
        stage_names = [transitions[0][0]] + [t[1] for t in transitions]
        stage_sizes = [transitions[0][2].sum(axis=1).astype(float)] + [
            t[2].sum(axis=0).astype(float) for t in transitions
        ]
        stage_tops = [_stack(sizes) for sizes in stage_sizes]

        # Consistent colour per family across all columns.
        names_all = list(dict.fromkeys(n for names in stage_names for n in names))
        if colourmap is not None:
            cmap = plt.get_cmap(colourmap)
            colours = {name: cmap(i % cmap.N) for i, name in enumerate(names_all)}
        else:
            colours = {name: _colour_for(name, i) for i, name in enumerate(names_all)}

        # Node column edges: the first/last columns sit flush with the plot
        # edges (as in the two-stage case); interior columns are centred on
        # their evenly spaced x position.
        x_positions = np.linspace(0.0, 1.0, n_stages)
        x_node_left = np.empty(n_stages)
        x_node_right = np.empty(n_stages)
        for k, x in enumerate(x_positions):
            if k == 0:
                x_node_left[k], x_node_right[k] = 0.0, node_width
            elif k == n_stages - 1:
                x_node_left[k], x_node_right[k] = 1.0 - node_width, 1.0
            else:
                x_node_left[k] = x - 0.5 * node_width
                x_node_right[k] = x + 0.5 * node_width

        # Ribbons: one pass per consecutive stage pair, outer loop over
        # source, inner over target, so each node's attachment points fill
        # top-down in a consistent order.
        for k, (row_names, col_names, matrix) in enumerate(transitions):
            x_left = x_node_right[k]  # right edge of the source column
            x_right = x_node_left[k + 1]  # left edge of the target column
            x_ctrl = 0.5 * (x_left + x_right)  # Bezier control x (smooth S-curve)
            left_cursor = stage_tops[k].copy()
            right_cursor = stage_tops[k + 1].copy()
            for i in range(len(row_names)):
                for j in range(len(col_names)):
                    flow = matrix[i, j]
                    if flow <= 0:
                        continue
                    yl_top, yl_bot = left_cursor[i], left_cursor[i] - flow
                    yr_top, yr_bot = right_cursor[j], right_cursor[j] - flow
                    vertices = [
                        (x_left, yl_top),
                        (x_ctrl, yl_top),
                        (x_ctrl, yr_top),
                        (x_right, yr_top),
                        (x_right, yr_bot),
                        (x_ctrl, yr_bot),
                        (x_ctrl, yl_bot),
                        (x_left, yl_bot),
                        (x_left, yl_top),
                    ]
                    codes = [
                        Path.MOVETO,
                        Path.CURVE4,
                        Path.CURVE4,
                        Path.CURVE4,
                        Path.LINETO,
                        Path.CURVE4,
                        Path.CURVE4,
                        Path.CURVE4,
                        Path.CLOSEPOLY,
                    ]
                    ax.add_patch(
                        PathPatch(
                            Path(vertices, codes),
                            facecolor=colours[row_names[i]],
                            edgecolor="none",
                            alpha=alpha,
                        )
                    )
                    left_cursor[i] = yl_bot
                    right_cursor[j] = yr_bot

        # Node bars. The first/last columns also get an external name+count
        # label (as in the two-stage case); interior columns rely on the
        # shared legend below, since ribbons on both sides leave no clear
        # spot for text.
        for k in range(n_stages):
            for name, size, top in zip(stage_names[k], stage_sizes[k], stage_tops[k]):
                ax.add_patch(
                    Rectangle(
                        (x_node_left[k], top - size),
                        x_node_right[k] - x_node_left[k],
                        size,
                        color=colours[name],
                    )
                )
                if k == 0:
                    ax.text(
                        x_node_left[k] - 0.01,
                        top - 0.5 * size,
                        f"{_latex_label(name)} ({_format_total(size)})",
                        ha="right",
                        va="center",
                    )
                elif k == n_stages - 1:
                    ax.text(
                        x_node_right[k] + 0.01,
                        top - 0.5 * size,
                        f"{_latex_label(name)} ({_format_total(size)})",
                        ha="left",
                        va="center",
                    )

        y_top = max(tops[0] for tops in stage_tops)
        y_bot = min(
            (tops[-1] - sizes[-1]).item()
            for tops, sizes in zip(stage_tops, stage_sizes)
        )
        for k in range(n_stages):
            x_centre = 0.5 * (x_node_left[k] + x_node_right[k])
            ax.text(x_centre, y_top + gap, stage_labels[k], ha="center", va="bottom")

        if n_stages > 2:
            from matplotlib.patches import Patch

            handles = [
                Patch(facecolor=colours[name], label=_latex_label(name))
                for name in names_all
            ]
            ax.legend(
                handles=handles,
                title="orbit class",
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=min(len(handles), 4),
            )

        margin = 0.05 * (y_top - y_bot)
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(y_bot - margin, y_top + gap + margin)
        ax.axis("off")
        return ax


def _named_resonance_vectors(max_order: int) -> List[tuple]:
    """Resonance vectors of the named boxlets above ``max_order``.

    For every plane (i, j) and every named ratio p:q (:data:`NAMED_BOXLETS`)
    both orientations, ``w_i:w_j = p:q`` and ``q:p``, give the vectors
    ``(q, -p)`` and ``(p, -q)`` on axes (i, j) (canonical sign: first non-zero
    component positive).

    Parameters
    ----------
    max_order : int
        Only vectors of L1 order above this are returned (the lower orders are
        already in the generic search).

    Returns
    -------
    vectors : list of tuple
        ``(nx, ny, nz)`` integer triples.
    """
    out = []
    for p, q in NAMED_BOXLETS:
        if p + q <= max_order:
            continue
        for i, j in ((0, 1), (0, 2), (1, 2)):
            for a, b in ((q, -p), (p, -q)):
                n = [0, 0, 0]
                n[i], n[j] = a, b
                out.append(tuple(n))
    return out


def _primitive_resonance_vectors(max_order: int, extra: Iterable[tuple] = ()):
    """Primitive integer triples n, canonical sign, ordered by L1 order.

    The bound is on the L1 order ``|n_x| + |n_y| + |n_z|`` -- the physically
    meaningful resonance order -- not on the individual components, so only
    genuinely low-order commensurabilities (banana 2:-1:0, fish, pretzel, ...)
    qualify.

    Parameters
    ----------
    max_order : int
        Maximum L1 order ``|n|_1`` of the resonance vectors to generate.
    extra : iterable of tuple, optional
        Further (primitive, canonically signed) vectors of any order to
        include, e.g. :func:`_named_resonance_vectors`.

    Returns
    -------
    vecs : numpy.ndarray
        (M, 3) float array of primitive resonance vectors, sorted by order.
    orders : numpy.ndarray
        (M,) L1 order of each vector, ascending.
    """
    out = []
    rng = range(-max_order, max_order + 1)
    for nx in rng:
        for ny in rng:
            for nz in rng:
                order = abs(nx) + abs(ny) + abs(nz)
                if order == 0 or order > max_order:
                    continue
                if gcd(gcd(abs(nx), abs(ny)), abs(nz)) != 1:
                    continue
                first = nx if nx != 0 else (ny if ny != 0 else nz)
                if first < 0:  # canonical sign
                    continue
                out.append((order, (nx, ny, nz)))
    known = {v for _, v in out}
    for v in extra:
        if tuple(v) not in known:
            out.append((sum(abs(c) for c in v), tuple(v)))
            known.add(tuple(v))
    out.sort(key=lambda t: t[0])
    orders = np.array([o for o, _ in out], dtype=np.int64)
    vecs = np.array([v for _, v in out], dtype=np.float64)  # (M,3)
    return vecs, orders


def _lattice_vectors(max_order: int) -> np.ndarray:
    """All non-zero integer triples with L1 order up to ``max_order``.

    Unlike :func:`_primitive_resonance_vectors`, this returns the full integer
    lattice (both signs, non-primitive included) used to test whether a spectral
    line is an integer *combination* of the base frequencies.

    Parameters
    ----------
    max_order : int
        Maximum L1 order ``|n_x| + |n_y| + |n_z|`` of the vectors.

    Returns
    -------
    vecs : numpy.ndarray
        (M, 3) float array of integer combination vectors.
    """
    rng = range(-max_order, max_order + 1)
    out = [
        (nx, ny, nz)
        for nx in rng
        for ny in rng
        for nz in rng
        if 1 <= abs(nx) + abs(ny) + abs(nz) <= max_order
    ]
    return np.array(out, dtype=np.float64)


def _lattice_vectors_2d(max_order: int) -> np.ndarray:
    """Non-zero integer pairs with L1 order up to ``max_order`` (see below).

    The two-base analogue of :func:`_lattice_vectors`, used when reducing a line
    to a combination of the first two base frequencies.

    Parameters
    ----------
    max_order : int
        Maximum L1 order ``|n_0| + |n_1|`` of the vectors.

    Returns
    -------
    vecs : numpy.ndarray
        (P, 2) float array of integer combination vectors.
    """
    rng = range(-max_order, max_order + 1)
    out = [(a, b) for a in rng for b in rng if 1 <= abs(a) + abs(b) <= max_order]
    return np.array(out, dtype=np.float64)


def _resolve_n_workers(n_workers: Optional[int]) -> int:
    """Resolve a worker count for the classification chunk thread pools.

    Defaults to serial (``1``): the dominant reduction (:func:`_detect_irregular`)
    is memory-bandwidth-bound over large broadcast temporaries, and profiling on
    real multi-million-orbit data showed a thread pool adds contention rather
    than throughput -- every thread count tried (4/8/16) was slower than serial,
    getting worse as threads increased. Pass ``n_workers`` explicitly to opt in
    and measure it on your own workload/hardware, where it may behave
    differently.
    """
    if n_workers is not None:
        return max(1, int(n_workers))
    return 1


def _map_chunks(n, chunk, worker, n_workers):
    """Apply ``worker(lo, hi)`` to disjoint ``[lo, hi)`` ranges covering ``[0, n)``.

    Runs serially if there is only one chunk or ``n_workers <= 1``. Otherwise
    the (independent, read-only-input) chunks are dispatched to a thread pool:
    numpy releases the GIL for the large elementwise/matmul work each chunk
    does here, so this scales across cores despite the GIL.

    Returns
    -------
    results : list of (lo, hi, value)
        One entry per chunk, in order, where ``value`` is ``worker(lo, hi)``.
    """
    ranges = [(lo, min(lo + chunk, n)) for lo in range(0, n, chunk)]
    if n_workers <= 1 or len(ranges) <= 1:
        return [(lo, hi, worker(lo, hi)) for lo, hi in ranges]
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [(lo, hi, pool.submit(worker, lo, hi)) for lo, hi in ranges]
        return [(lo, hi, fut.result()) for lo, hi, fut in futures]


def _detect_irregular(fundamentals, lines, amp_frac, tol, max_order, n_workers=None):
    """Flag orbits needing more than three base frequencies (irregular).

    Following Frigo et al. (2021) / Carpintero & Aguilar (1998), an orbit is
    *irregular* when its spectrum cannot be described by three base frequencies.
    The base frequencies are identified greedily from the reduced spectral lines:
    the strongest *significant* line (amplitude at least ``amp_frac`` of the
    orbit's strongest line) is the first base; the strongest line that is not an
    integer combination of the bases so far becomes the next base; and so on. If
    a fourth independent base frequency is found the orbit is irregular. A line
    is an integer combination when it lies within ``tol * max|w_fund|`` of
    ``n_0 b_0 + n_1 b_1 + n_2 b_2`` for integers with ``|n|_1 <= max_order``.

    Parameters
    ----------
    fundamentals : numpy.ndarray
        (N, 3) signed fundamental frequency per axis (used for the frequency
        scale).
    lines : numpy.ndarray
        (N, 3, n_lines, 2) leading (frequency, amplitude) spectral lines.
    amp_frac : float
        A line counts as significant if its amplitude is at least this fraction
        of the orbit's largest line amplitude.
    tol : float
        Relative tolerance for the integer-combination test.
    max_order : int
        Maximum L1 order of the integer combinations tested.
    n_workers : int, optional
        Number of threads to process chunks with; chunks are independent, but
        this reduction is memory-bandwidth-bound, so threading was measured to
        *hurt* on real data rather than help (see :func:`_resolve_n_workers`).
        Defaults to serial (``1``); pass this explicitly only if you've
        measured a benefit on your own workload.

    Returns
    -------
    irregular : numpy.ndarray
        (N,) boolean, True where the orbit is irregular (> 3 base frequencies).
    """
    fundamentals = np.asarray(fundamentals)
    lines = np.asarray(lines)
    # Work in the results' own float dtype (float32 once round-tripped through
    # save/load) rather than forcing float64: this reduction is the dominant
    # cost of classification and is elementwise/bandwidth-bound, so narrowing
    # the dtype roughly halves it, and the tolerances used here don't need
    # float64 precision.
    dtype = np.result_type(fundamentals.dtype, lines.dtype)
    f = np.abs(fundamentals).astype(dtype, copy=False)  # (N,3)
    freqs = lines[..., 0].astype(dtype, copy=False)  # (N,3,nl)
    amps = lines[..., 1].astype(dtype, copy=False)  # (N,3,nl)
    n = f.shape[0]
    cols = freqs.shape[1] * freqs.shape[2]
    line_abs = np.abs(freqs.reshape(n, cols))  # (N,C) magnitudes
    line_amp = amps.reshape(n, cols)  # (N,C)

    scale = np.maximum(np.max(f, axis=1), 1e-30)  # (N,) frequency scale
    amp_max = np.maximum(np.max(line_amp, axis=1), 1e-30)  # (N,)

    mult1 = np.arange(1, max_order + 1, dtype=dtype)  # (K,) single-base
    vecs2 = _lattice_vectors_2d(max_order).astype(dtype, copy=False)  # (P,2)
    vecs3 = _lattice_vectors(max_order).astype(dtype, copy=False)  # (Q,3)

    def _reducible(la_abs, combos, tol_abs):
        # Nearest integer-combination distance per line <= tolerance.
        resid = np.abs(la_abs[:, :, None] - combos[:, None, :])  # (m, C, n_combos)
        return resid.min(axis=2) <= tol_abs  # (m, C)

    def _process(lo, hi):
        la = line_abs[lo:hi]  # (m, C)
        amp = line_amp[lo:hi]  # (m, C)
        tol_abs = tol * scale[lo:hi, None]  # (m, 1)
        rows = np.arange(hi - lo)
        significant = (amp >= amp_frac * amp_max[lo:hi, None]) & (la > tol_abs)

        def _pick(candidate):
            # Frequency of the strongest candidate line per orbit (0 if none).
            present = candidate.any(axis=1)
            idx = np.argmax(np.where(candidate, amp, -1.0), axis=1)
            return np.where(present, la[rows, idx], 0.0), present

        # First base: the strongest significant line.
        b0, has0 = _pick(significant)

        # Second base: strongest line not a multiple of b0.
        red0 = _reducible(la, b0[:, None] * mult1[None, :], tol_abs)
        b1, has1 = _pick(significant & ~red0)

        # Third base: strongest line not a combination of {b0, b1}.
        combos2 = np.stack([b0, b1], axis=1) @ vecs2.T  # (m, P)
        red1 = _reducible(la, combos2, tol_abs)
        b2, has2 = _pick(significant & ~red1)

        # Irregular: a fourth independent line beyond {b0, b1, b2}.
        combos3 = np.stack([b0, b1, b2], axis=1) @ vecs3.T  # (m, Q)
        red2 = _reducible(la, combos3, tol_abs)
        return has2 & np.any(significant & ~red2, axis=1)

    chunk = max(256, int(5_000_000 // max(1, cols * len(vecs3))))
    irregular = np.zeros(n, dtype=bool)
    for lo, hi, result in _map_chunks(
        n, chunk, _process, _resolve_n_workers(n_workers)
    ):
        irregular[lo:hi] = result
    return irregular


def _find_resonances(w, max_order, tol, chunk=20000, n_workers=None, extra=()):
    """Find the lowest-order commensurabilities of each frequency triple.

    A commensurability is a low-order integer vector ``n`` with
    ``|n . w| / max|w| < tol``. Besides the lowest-order one, the
    lowest-order *second* commensurability of order >= 2 independent of the
    first is returned: an orbit with two independent resonances is a closed
    (periodic) orbit, not a thin one.

    Parameters
    ----------
    w : numpy.ndarray
        (N, 3) absolute fundamental frequencies per orbit.
    max_order : int
        Maximum L1 order of the resonance vectors to consider.
    tol : float
        Relative tolerance ``|n . w| / max|w|`` for accepting a resonance.
    chunk : int, optional
        Number of orbits processed per block (bounds memory).
    n_workers : int, optional
        Number of threads to process chunks with; see :func:`_detect_irregular`.
    extra : iterable of tuple, optional
        Further resonance vectors searched for beyond ``max_order`` (see
        :func:`_primitive_resonance_vectors`).

    Returns
    -------
    vectors : numpy.ndarray
        (N, 3) integer resonance vector per orbit (all zero if none found).
    orders : numpy.ndarray
        (N,) L1 order of each resonance (0 if none found).
    second_vectors : numpy.ndarray
        (N, 3) second, independent resonance vector of order >= 2 (all zero
        if none).
    second_orders : numpy.ndarray
        (N,) its L1 order (0 if none).
    """
    w = np.asarray(w)
    vecs, orders = _primitive_resonance_vectors(max_order, extra)
    vecs = vecs.astype(w.dtype, copy=False)  # match w's dtype (see _detect_irregular)
    N = len(w)
    scale = np.maximum(np.max(np.abs(w), axis=1), 1e-30)  # (N,)

    def _process(lo, hi):
        # |n . w| / max|w| for every candidate; vecs are sorted by order, so the
        # first hit is the lowest-order commensurability.
        resid = np.abs(w[lo:hi] @ vecs.T)  # (m, M)
        norm = resid / scale[lo:hi, None]
        hit = norm < tol
        any_hit = hit.any(axis=1)
        first = np.argmax(hit, axis=1)  # first (lowest order)
        rv = np.zeros((hi - lo, 3), dtype=np.int64)
        ro = np.zeros(hi - lo, dtype=np.int64)
        rv[any_hit] = vecs[first[any_hit]].astype(np.int64)
        ro[any_hit] = orders[first[any_hit]]
        # Second resonance: any other hit of order >= 2. Canonical primitive
        # vectors are parallel only if identical, so excluding the first
        # column leaves the independent ones.
        hit[np.arange(hi - lo), first] = False
        hit &= orders[None, :] >= 2
        any_second = hit.any(axis=1)
        second = np.argmax(hit, axis=1)
        sv = np.zeros((hi - lo, 3), dtype=np.int64)
        so = np.zeros(hi - lo, dtype=np.int64)
        sv[any_second] = vecs[second[any_second]].astype(np.int64)
        so[any_second] = orders[second[any_second]]
        return rv, ro, sv, so

    res_vec = np.zeros((N, 3), dtype=np.int64)
    res_ord = np.zeros(N, dtype=np.int64)
    sec_vec = np.zeros((N, 3), dtype=np.int64)
    sec_ord = np.zeros(N, dtype=np.int64)
    for lo, hi, (rv, ro, sv, so) in _map_chunks(
        N, chunk, _process, _resolve_n_workers(n_workers)
    ):
        res_vec[lo:hi] = rv
        res_ord[lo:hi] = ro
        sec_vec[lo:hi] = sv
        sec_ord[lo:hi] = so
    return res_vec, res_ord, sec_vec, sec_ord


def _describe_boxlets(resonance, w, is_boxlet):
    """Type, plane and elongation axis of each boxlet's resonance.

    Parameters
    ----------
    resonance : numpy.ndarray
        (N, 3) primitive resonance vector (canonical sign).
    w : numpy.ndarray
        (N, 3) absolute fundamental frequencies.
    is_boxlet : numpy.ndarray
        (N,) bool, the orbits to describe.

    Returns
    -------
    boxlet_type : numpy.ndarray
        (N,) :class:`BoxletType` values (NONE where not a boxlet).
    plane : numpy.ndarray
        (N,) axis the resonant plane is perpendicular to (the axis not in a
        two-axis resonance: 2 for an x-y resonance); -1 for spatial and
        non-boxlets.
    elongation : numpy.ndarray
        (N,) axis the boxlet is elongated along: the lower-frequency axis of a
        two-axis resonance, the lowest-frequency axis of a spatial one; -1 for
        non-boxlets.
    """
    n = len(resonance)
    boxlet_type = np.full(n, BoxletType.NONE, dtype=np.int64)
    plane = np.full(n, -1, dtype=np.int64)
    elongation = np.full(n, -1, dtype=np.int64)
    zeros = resonance == 0
    planar = is_boxlet & (zeros.sum(axis=1) == 1)
    spatial = is_boxlet & (zeros.sum(axis=1) == 0)
    boxlet_type[spatial] = BoxletType.SPATIAL
    elongation[spatial] = np.argmin(w[spatial], axis=1)
    for k in np.flatnonzero(planar):
        normal = int(np.flatnonzero(zeros[k])[0])
        i, j = [a for a in range(3) if a != normal]
        # n_i w_i = -n_j w_j: w_i:w_j = |n_j|:|n_i|.
        ratio = (abs(int(resonance[k, j])), abs(int(resonance[k, i])))
        boxlet_type[k] = NAMED_BOXLETS.get(tuple(sorted(ratio)), BoxletType.PLANAR)
        plane[k] = normal
        elongation[k] = i if w[k, i] <= w[k, j] else j
    return boxlet_type, plane, elongation


def classify_orbits(
    results,
    circ_thresh: float = 0.9,
    hole_thresh: float = 0.03,
    freq_tol: float = 0.01,
    amp_frac: float = 0.05,
    resonance_max_order: int = 5,
    resonance_tol: float = 1e-3,
    shared_line_tol: float = 0.1,
    inner_outer_ratio: Optional[float] = None,
    irregular_amp_frac: float = 0.1,
    irregular_tol: float = 0.02,
    irregular_max_order: int = 6,
    diffusion_threshold: Optional[float] = 0.1,
    diffusion_drop: float = 0.5,
    drop_binary_interacting: bool = False,
    drop_runaways: bool = True,
    pattern_lock_tol: Optional[float] = 1e-3,
    libration_hole: float = 0.1,
    winding_thresh: float = 0.9,
    pattern_tol: float = 1.0,
    pattern_cycles: float = 3.0,
    resolution_cycles: float = 3.0,
    corotation_band: Tuple[float, float] = DEFAULT_COROTATION_BAND,
    family_table=FAMILY_TABLE,
) -> OrbitClassification:
    """Classify the orbits in an :class:`~lanfear.OrbitResults`.

    Parameters
    ----------
    results : lanfear.OrbitResults
        Orbits to classify. Must carry frequency data (``fundamentals``, and
        ideally ``lines``), as produced by :func:`analyse_family`; it drives the
        1:1:1 rosette and boxlet-resonance tests.
    circ_thresh : float, optional
        ``circ_a`` above this counts as circulation about axis ``a`` (a tube;
        default 0.9, calibrated on static and tumbling perfect ellipsoids, see
        :doc:`tuning`).
    hole_thresh : float, optional
        For a rotating figure, the minimum hole about the pattern-speed axis
        (:meth:`OrbitResults.hole_fraction`, closest approach to the axis over
        the mean radius) for circulation about it to count (default 0.03).
        Keeps out boxes, which precess steadily in a tumbling figure, and so
        can exceed ``circ_thresh``, but pass close to the centre. Not applied
        to a static potential or about the other axes (where it only removes
        thin tubes).
    freq_tol : float, optional
        A loop whose (amplitude-active) axis fundamentals are mutually equal to
        within this fraction is a rosette (1:1:1); a tube is 1:1:pi (two axes
        share a frequency, the circulation axis differs).
    amp_frac : float, optional
        An axis whose leading spectral amplitude exceeds this fraction of the
        largest is "active" (used to ignore silent axes in the 1:1:1 test).
    resonance_max_order : int, optional
        Maximum L1 order searched for the boxlet commensurability (default 5;
        the named boxlets of higher order, i.e. the pretzel, are searched for
        as well). Also used to corroborate the intermediate-axis (y) tube
        branch (see Notes).
    resonance_tol : float, optional
        Tolerance ``|n.w| / max|w|`` for accepting a boxlet resonance (and,
        equally, the y-tube corroboration above). The default ``1e-3`` keeps
        chance commensurabilities of non-resonant boxes to a few per cent
        (see the module notes); ``1e-2`` lets ~30% of them through.
    shared_line_tol : float, optional
        For a rotating figure, an orbit whose two fundamentals in the plane of
        rotation differ by less than this many frequency resolutions
        (:attr:`OrbitResults.frequency_resolution`) is never a boxlet (default
        0.1). The frame rotation mixes the two in-plane coordinates of a box
        that precesses about the pattern-speed axis, so both spectra are led by
        one and the same line; their equality is the precession, not a 1:1
        resonance. Independent oscillations do not agree that closely (on
        static models fewer than 0.5% of boxes do). 0 disables the check;
        ignored for a static potential.
    inner_outer_ratio : float or None, optional
        Long-axis (x) tubes whose ratio of long-axis to circulation frequency,
        ``|w_x| / sqrt(|w_y| |w_z|)``, is below this are *inner*, the others
        *outer* (outer tubes sit at ~0.95 for any shape; inner ones lower, by a
        shape-dependent amount). None (default) places the cut from the
        population's own long-axis tubes (:func:`adaptive_inner_outer_cut`),
        falling back to :data:`DEFAULT_INNER_OUTER_RATIO` (0.84) with a warning
        when the population does not look like a mix of the two families
        (e.g. too few long-axis tubes). The cut used is recorded in
        :attr:`OrbitClassification.inner_outer_ratio`. Because an adaptive cut
        depends on the population, pass a number to classify several
        populations (e.g. snapshots to be compared) with the same cut.
    irregular_amp_frac : float, optional
        Amplitude threshold (as a fraction of an orbit's strongest line) above
        which a spectral line is tested for the irregular criterion. Requires
        frequency data.
    irregular_tol : float, optional
        Relative tolerance ``|w - n.w_fund| / max|w_fund|`` for deciding whether
        a line is an integer combination of the three fundamentals.
    irregular_max_order : int, optional
        Maximum L1 order of the integer combinations tested for the irregular
        criterion.
    diffusion_threshold : float or None, optional
        An orbit whose Laskar frequency-diffusion rate
        (:meth:`OrbitResults.diffusion_rate`) exceeds this value is labelled
        irregular (chaotic), in addition to the spectral criterion above. The
        default ``0.1`` favours purity: on tumbling perfect ellipsoids it flags
        ~1% of the regular orbits but only ~15--30% of the chaotic ones
        (Lyapunov reference, see :doc:`tuning`), so the irregular fraction is a
        lower bound. Lower values catch more chaos at the cost of flagging more
        regular orbits (e.g. 0.03: ~50% / ~2.5%). The separation depends on the
        integration length; inspect ``np.log10(results.diffusion_rate())`` (or
        run ``scripts/choose_diffusion_threshold.py``). ``None`` disables the
        cut.
    diffusion_drop : float, optional
        For orbits that were re-integrated for longer (see ``max_extensions`` in
        :func:`lanfear.orbits.analyse_family`), the cut is applied to the
        *trend*: a regular orbit's drift falls as the window grows, so an
        extended orbit is irregular only if its rate is above the threshold *and*
        at least ``diffusion_drop`` times its rate before the last extension.
        An extended orbit whose drift is still above the threshold but falling
        keeps its regular label. Orbits never extended (including all orbits of a
        run without extension) use the plain threshold cut.
    drop_binary_interacting : bool, optional
        If True, orbits flagged by :attr:`OrbitResults.binary_interacting`
        (pericentre within the central SMBH binary) are removed before
        classifying, so the returned classification covers only the remaining
        orbits (see :meth:`OrbitResults.drop_binary_interacting`). Default
        False: every orbit is classified.
    drop_runaways : bool, optional
        If True (default), orbits flagged by :attr:`OrbitResults.runaway`
        (leaving the system within the integration) are removed before
        classifying, so the returned classification covers only the remaining
        orbits (see :meth:`OrbitResults.drop_runaways`); match it to other
        per-orbit data by :attr:`OrbitClassification.ids`. If False, every
        orbit is classified, with a warning if runaways are present.
    pattern_lock_tol : float or None, optional
        For a rotating figure, orbits flagged by
        :meth:`OrbitResults.pattern_locked` with this tolerance are never
        rosettes (their shared x/y frequency is the frame rotation). ``None``
        disables the check. Ignored for a static potential.
    libration_hole : float, optional
        For a rotating figure, a non-loop whose hole about the pattern-speed
        axis (:meth:`OrbitResults.hole_fraction`) exceeds this and whose
        azimuth about that axis spans less than ``2 pi`` (``phi_*_range``) is
        ``LIBRATING`` (default 0.1).
    winding_thresh : float, optional
        For a rotating figure, a box (non-loop) whose winding circulation about
        the pattern-speed axis (:meth:`OrbitResults.winding_circulation`)
        exceeds this is tested for pattern resonances (default 0.9). Bar
        orbits such as x1 go round the pattern axis in the body frame but pass
        close to the centre, so their shape is a box. Loops about another axis
        are not tested.
    pattern_tol : float, optional
        Tolerance of the pattern resonances, in frequency resolutions
        ``2 pi / t_total``: corotation if ``|Omega_phi,body|`` is below it, a
        ``l:1`` resonance if ``|Omega_R - l |Omega_phi,body||`` is (default 1).
    pattern_cycles : float, optional
        Minimum cycles of ``Omega_R`` and ``Omega_phi,body`` in the window for
        the 2:1 and 4:1 tests (default 3); fewer give ``UNRESOLVED``.
    resolution_cycles : float, optional
        Minimum cycles in the window of every frequency an orbit's labels
        depend on; orbits with fewer are flagged
        :attr:`OrbitClassification.unresolved` (default 3).
    corotation_band : tuple of float, optional
        ``(low, high)`` range of each orbit's ``epsilon = |Omega_p| /
        |Omega_phi|`` flagged by :attr:`OrbitClassification.near_corotation`
        (default :data:`DEFAULT_COROTATION_BAND`, ``(0.5, inf)``: at or beyond
        corotation, calibrated on tumbling perfect ellipsoids, see
        :doc:`tuning`). Ignored for a static figure.
    family_table : sequence of FamilyRule, optional
        Ordered rules naming each orbit's family from its labels (default
        :data:`FAMILY_TABLE`; see :func:`assign_families`).

    Returns
    -------
    classification : OrbitClassification
        Per-orbit family labels and the diagnostic quantities used to derive
        them (circulation, tube axis, planarity, resonance vector and order),
        the geometric shape, the sense and the boxlet subdivision.

    Notes
    -----
    An orbit whose spectrum needs more than
    three base frequencies is labelled :attr:`OrbitClass.IRREGULAR` (Frigo et
    al. 2021), overriding the regular-family assignment: such an orbit is not
    confined to a regular 3-torus and is a likely-chaotic candidate.

    An orbit whose dominant circulation is about y is only accepted as
    :attr:`OrbitClass.INTERMEDIATE_AXIS_TUBE` if its x and z fundamentals also
    share a 1:1 lock -- Carpintero & Aguilar's own frequency-domain definition
    of a y-tube. Without that lock it is instead labelled
    :attr:`OrbitClass.BOXLET` if it satisfies the boxlet commensurability test
    (the generic signature of a resonant box orbit elongated along the
    intermediate axis, e.g. Merritt & Valluri 1999; Poon & Merritt 2001), or
    :attr:`OrbitClass.PIBOX` otherwise. Either can mimic persistent L_y
    circulation over a finite integration window without being a genuine (and
    generically unstable) y-tube.
    """
    if drop_binary_interacting:
        results = results.drop_binary_interacting()
    if drop_runaways:
        results = results.drop_runaways()
    else:
        n_runaway = int(results.runaway.sum())
        if n_runaway:
            logger.warning(
                f"Classifying {n_runaway} runaway orbits (of {len(results.ids)}); "
                f"their labels are meaningless (drop_runaways=False was passed; "
                f"the default excludes them)"
            )
    c = results.column
    status = c("status")
    N = len(status)

    Lm = np.stack([c("Lx_mean"), c("Ly_mean"), c("Lz_mean")], axis=1)
    La = np.stack([c("Lx_abs_mean"), c("Ly_abs_mean"), c("Lz_abs_mean")], axis=1)
    circ = np.abs(Lm) / np.maximum(La, 1e-30)  # (N,3)

    # Rotating figure: the frame-aware tests need the pattern-speed axis and
    # the planar spectra / winding columns.
    pattern = _rotating_frame(results)

    circulating = circ > circ_thresh
    if pattern is not None:
        # A box precessing steadily about the pattern axis keeps one sense of
        # L there most of the time, but still passes close to the centre.
        with np.errstate(invalid="ignore"):
            circulating[:, pattern] &= results.hole_fraction()[:, pattern] > hole_thresh
    ok = status == 0
    is_loop = circulating.any(axis=1)
    tube_axis = np.where(
        is_loop,
        np.argmax(np.where(circulating, circ, -1.0), axis=1),
        np.argmax(circ, axis=1),
    )

    # Shape tensor -> planarity (smallest / largest eigenvalue).
    S = np.zeros((N, 3, 3))
    S[:, 0, 0] = c("Sxx")
    S[:, 1, 1] = c("Syy")
    S[:, 2, 2] = c("Szz")
    S[:, 0, 1] = S[:, 1, 0] = c("Sxy")
    S[:, 0, 2] = S[:, 2, 0] = c("Sxz")
    S[:, 1, 2] = S[:, 2, 1] = c("Syz")
    evals = np.linalg.eigvalsh(S)  # ascending (N,3)
    planarity = evals[:, 0] / np.maximum(evals[:, 2], 1e-30)

    # Rosette test: are the active-axis fundamentals mutually 1:1:1?
    w = np.abs(results.fundamentals)  # (N,3)
    # In a rotating figure, the in-plane fundamentals of a loop about the
    # pattern-speed axis are shifted by the frame rotation; compare the
    # inertial azimuthal frequency (exact in that plane) with the out-of-plane
    # one, which the rotation leaves alone.
    w_rosette = w.copy()
    if pattern is not None:
        a = pattern
        about = is_loop & (tube_axis == a)
        omega_in = np.abs(results.azimuthal_frequency("inertial")[:, a])
        for i in range(3):
            if i != a:
                w_rosette[about, i] = omega_in[about]
    amp = results.lines[:, :, 0, 1]
    active = amp > amp_frac * np.max(amp, axis=1, keepdims=True)
    n_active = np.sum(active, axis=1)
    w_hi = np.where(active, w_rosette, -np.inf).max(axis=1)
    w_lo = np.where(active, w_rosette, np.inf).min(axis=1)
    freq_111 = (n_active >= 2) & (w_hi <= (1.0 + freq_tol) * np.maximum(w_lo, 1e-30))
    if pattern is not None and pattern_lock_tol is not None:
        # The shared |Omega_p| of a pattern-locked orbit is the frame
        # rotation, not a 1:1:1 symmetry of the orbit.
        freq_111 &= ~results.pattern_locked(pattern_lock_tol)
    res_vec, res_ord, sec_vec, sec_ord = _find_resonances(
        w,
        resonance_max_order,
        resonance_tol,
        extra=_named_resonance_vectors(resonance_max_order),
    )
    resonant = res_ord >= 2
    if pattern is not None:
        # A box precessing about the pattern axis has one shared line in both
        # in-plane spectra: that equality is the frame rotation, not a 1:1
        # resonance.
        i, j = [k for k in range(3) if k != pattern]
        with np.errstate(invalid="ignore"):
            shared_line = np.abs(w[:, i] - w[:, j]) < (
                shared_line_tol * results.frequency_resolution
            )
        resonant &= ~shared_line
    # Frigo/Carpintero & Aguilar's own frequency-domain definition of a y-tube
    # is an x:z lock (the two axes NOT circulated about share a 1:1 resonance).
    xz_locked = np.abs(w[:, 0] - w[:, 2]) <= freq_tol * np.maximum(
        np.maximum(w[:, 0], w[:, 2]), 1e-30
    )

    labels = np.full(N, OrbitClass.UNCLASSIFIED, dtype=np.int64)

    # Rosette: a loop whose axis fundamentals are all commensurate 1:1:1.
    labels[ok & is_loop & freq_111] = OrbitClass.ROSETTE

    # Tubes: a single circulation axis, not 1:1:1.
    tube = ok & is_loop & ~freq_111
    labels[tube & (tube_axis == 2)] = OrbitClass.SHORT_AXIS_TUBE
    # Intermediate-axis (y) circulation is the one family that is generically
    # unstable in a triaxial potential (Heiligman & Schwarzschild 1979), so a
    # high circ_y over a finite integration window is also what a resonant box
    # orbit elongated along y looks like (e.g. the (1,-2,1) family of Merritt &
    # Valluri 1999, seen near a central mass by Poon & Merritt 2001). Corroborate
    # the circulation-based candidate two ways before accepting it:
    #  1. Frigo/Carpintero & Aguilar's own frequency-domain y-tube test: the
    #     two non-circulating coordinates (x, z) must share a 1:1 lock. A
    #     locked pair is the tube's own signature, not boxlet evidence (two
    #     equal frequencies trivially register as a low-order commensurability
    #     in the general resonance search), so it is accepted outright.
    #  2. Failing that lock, check the general boxlet resonance test: a hit
    #     means "resonant box" (e.g. (1,-2,1)), not "y-tube".
    # An unlocked, non-resonant candidate has neither confirmation and falls
    # back to an unconfirmed (plain) box -- mirroring Frigo's own fall-through
    # when no pairwise commensurability is found at all.
    y_tube = tube & (tube_axis == 1)
    y_locked = y_tube & xz_locked
    y_resonant = y_tube & ~xz_locked & resonant
    labels[y_locked] = OrbitClass.INTERMEDIATE_AXIS_TUBE
    labels[y_resonant] = OrbitClass.BOXLET
    labels[y_tube & ~xz_locked & ~resonant] = OrbitClass.PIBOX
    # Long-axis (x) tubes split inner/outer by how fast they oscillate along
    # the long axis compared with their 1:1 circulation in the y-z plane:
    # |w_x| / sqrt(|w_y| |w_z|) is ~0.95 for outer tubes and lower for inner
    # ones, by a shape-dependent amount, so by default the cut is placed from
    # the population itself (adaptive_inner_outer_cut).
    lat = tube & (tube_axis == 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        lat_ratio = w[:, 0] / np.sqrt(w[:, 1] * w[:, 2])
    adaptive = inner_outer_ratio is None
    if adaptive:
        inner_outer_ratio, reason = adaptive_inner_outer_cut(lat_ratio[lat])
        if inner_outer_ratio is None:
            adaptive = False
            inner_outer_ratio = DEFAULT_INNER_OUTER_RATIO
            if np.any(lat):
                logger.warning(
                    f"No adaptive inner/outer long-axis-tube cut ({reason}); using "
                    f"the fixed {DEFAULT_INNER_OUTER_RATIO} (calibrated on a triaxial "
                    f"figure; pass inner_outer_ratio to set it)"
                )
        else:
            logger.info(
                f"Adaptive inner/outer long-axis-tube cut: {inner_outer_ratio:.3f}"
            )
    with np.errstate(invalid="ignore"):
        inner = lat & (lat_ratio < inner_outer_ratio)
    labels[inner] = OrbitClass.INNER_LONG_AXIS_TUBE
    labels[lat & ~inner] = OrbitClass.OUTER_LONG_AXIS_TUBE

    # Boxes: no circulation. A low-order (>=2) resonance marks a boxlet.
    box = ok & ~is_loop
    labels[box] = OrbitClass.PIBOX
    labels[box & resonant] = OrbitClass.BOXLET

    # Geometric shape and boxlet type, before the irregular override hides them.
    shape = np.full(N, OrbitShape.UNCLASSIFIED, dtype=np.int64)
    for cls, shp in _CLASS_TO_SHAPE.items():
        shape[labels == cls] = shp
    is_boxlet = labels == OrbitClass.BOXLET
    boxlet_type, boxlet_plane, boxlet_elongation = _describe_boxlets(
        res_vec, w, is_boxlet
    )
    boxlet_closed = is_boxlet & (sec_ord >= 2)

    # Sense of motion: the sign of the azimuthal frequency about the loop's
    # circulation axis (inertial about the pattern-speed axis).
    loop = np.isin(shape, _LOOP_SHAPES)
    frequency = results.azimuthal_frequency()
    if pattern is not None:
        frequency[:, pattern] = results.azimuthal_frequency("inertial")[:, pattern]
    about_axis = frequency[np.arange(N), tube_axis]
    sense = np.full(N, OrbitSense.NONE, dtype=np.int64)
    sense[loop & (about_axis > 0)] = OrbitSense.PROGRADE
    sense[loop & (about_axis < 0)] = OrbitSense.RETROGRADE
    if pattern is not None:
        hole = results.hole_fraction()[:, pattern]
        spread = np.asarray(c(f"phi_{'xyz'[pattern]}_range"), dtype=np.float64)
        with np.errstate(invalid="ignore"):
            librating = ok & ~loop & (hole > libration_hole) & (spread < 2 * np.pi)
        sense[librating] = OrbitSense.LIBRATING
        if pattern_lock_tol is not None:
            sense[ok & results.pattern_locked(pattern_lock_tol)] = OrbitSense.LOCKED

    # Resonances with the pattern, for orbits going round its axis: loops
    # about it, and boxes winding steadily about it (bar orbits such as x1);
    # loops about another axis are not tested.
    pattern_resonance = np.full(N, PatternResonance.NONE, dtype=np.int64)
    pattern_ratio = np.full(N, np.nan)
    if pattern is not None:
        a = pattern
        with np.errstate(invalid="ignore"):
            winding = results.winding_circulation()[:, a] > winding_thresh
        eligible = (
            ok
            & (
                (loop & (tube_axis == a))
                | (~loop & winding)
                | (sense == OrbitSense.LIBRATING)
            )
            & (sense != OrbitSense.LOCKED)
        )
        omega_body = results.azimuthal_frequency()[:, a]
        omega_inertial = results.azimuthal_frequency("inertial")[:, a]
        omega_r = np.asarray(c("radial_frequency"), dtype=np.float64)
        tolerance = pattern_tol * results.frequency_resolution
        with np.errstate(invalid="ignore", divide="ignore"):
            at_corotation = eligible & (np.abs(omega_body) <= tolerance)
            # Corotation is only distinguishable from no rotation if the
            # pattern itself completes enough cycles in the window.
            pattern_resolved = results.resolved(
                np.full(N, results.pattern_speed_ho[a]), pattern_cycles
            )
            corotation = at_corotation & pattern_resolved
            resolved = results.resolved(omega_r, pattern_cycles) & results.resolved(
                omega_body, pattern_cycles
            )
            tested = eligible & ~at_corotation & resolved
            pattern_ratio[tested] = omega_r[tested] / np.abs(omega_body[tested])
            two = tested & (np.abs(omega_r - 2 * np.abs(omega_body)) <= tolerance)
            four = (
                tested & ~two & (np.abs(omega_r - 4 * np.abs(omega_body)) <= tolerance)
            )
        # Elongation in the plane of rotation: along the lower-index (longer)
        # in-plane axis -> x1, across it -> x2.
        i, j = [k for k in range(3) if k != a]
        second_moment = [c(f"S{'xyz'[k]}{'xyz'[k]}") for k in (i, j)]
        along = second_moment[0] >= second_moment[1]
        pattern_resonance[
            (eligible & ~at_corotation & ~resolved)
            | (at_corotation & ~pattern_resolved)
        ] = PatternResonance.UNRESOLVED
        pattern_resonance[corotation] = PatternResonance.COROTATION
        prograde = omega_body > 0
        pattern_resonance[two & prograde & along] = PatternResonance.X1
        pattern_resonance[two & prograde & ~along] = PatternResonance.X2
        pattern_resonance[two & ~prograde & (omega_inertial < 0)] = PatternResonance.X4
        pattern_resonance[two & ~prograde & (omega_inertial >= 0)] = (
            PatternResonance.OLR
        )
        pattern_resonance[four] = PatternResonance.ULTRAHARMONIC
        counts = {
            PATTERN_RESONANCE_NAMES[int(v)]: int(n)
            for v, n in zip(*np.unique(pattern_resonance[eligible], return_counts=True))
        }
        logger.info(f"Pattern resonances of {int(eligible.sum())} orbits: {counts}")

    # Irregular (Frigo et al. 2021): a spectrum needing > 3 base frequencies.
    # Determined from the spectral lines, it overrides the regular-family label.
    irregular = _detect_irregular(
        results.fundamentals,
        results.lines,
        irregular_amp_frac,
        irregular_tol,
        irregular_max_order,
    )
    labels[ok & irregular] = OrbitClass.IRREGULAR
    regularity = np.full(N, Regularity.UNKNOWN, dtype=np.int64)
    regularity[ok] = Regularity.REGULAR
    regularity[ok & irregular] = Regularity.IRREGULAR

    # Chaotic by frequency diffusion (Laskar): drifting frequencies also
    # override the regular-family label. NaN rates compare False (not flagged).
    if diffusion_threshold is not None:
        with np.errstate(invalid="ignore"):
            rate = results.diffusion_rate()
            chaotic = ok & (rate > diffusion_threshold)
            previous = results.diffusion_previous
            if previous is not None:
                # Extended orbits: chaotic only if the drift did not fall.
                extended = np.isfinite(previous)
                chaotic &= ~extended | (rate >= diffusion_drop * previous)
        labels[chaotic] = OrbitClass.IRREGULAR
        regularity[chaotic] = Regularity.DIFFUSING

    # Reliability: does any label rest on a frequency completing too few cycles?
    unresolved = ok & np.any(active & ~results.resolved(w, resolution_cycles), axis=1)
    with np.errstate(invalid="ignore"):
        unresolved |= loop & ~results.resolved(about_axis, resolution_cycles)
    unresolved |= pattern_resonance == PatternResonance.UNRESOLVED

    pattern_speed_ratio = results.pattern_speed_ratio()
    if pattern is not None:
        low, high = corotation_band
        with np.errstate(invalid="ignore"):
            n_near = int(
                np.sum(
                    ok & (pattern_speed_ratio >= low) & (pattern_speed_ratio <= high)
                )
            )
        logger.info(
            f"{n_near} orbits ({100 * n_near / max(N, 1):.1f}%) are near or beyond "
            f"corotation ({low:g} <= |Omega_p| / |Omega_phi| <= {high:g}); their "
            f"labels are the least reliable (see OrbitClassification.near_corotation)"
        )

    # The class label is a view of the separate labels; families come from the
    # ordered rule table.
    labels = _legacy_labels(shape, boxlet_type, regularity)
    family, family_names = assign_families(
        {
            "shape": shape,
            "boxlet": boxlet_type,
            "sense": sense,
            "resonance": pattern_resonance,
            "regularity": regularity,
        },
        family_table,
    )
    counts = {
        CLASS_NAMES[int(v)]: int(cnt)
        for v, cnt in zip(*np.unique(labels, return_counts=True))
    }
    logger.info(f"Classified {N} orbits: {counts}")

    # Radii are reported in physical units. Both the snapshot radius and the
    # orbit-averaged radius r_mean are stored in HO units, so scale them by the
    # length unit (the scale radius). The snapshot radius is the default binning
    # radius for plot_class_fractions.
    length_unit = results.length_unit
    radius = np.asarray(results.initial_radius, dtype=float) * length_unit
    radius_orbit_averaged = c("r_mean") * length_unit

    # HO specific energy (the Jacobi integral for a rotating figure) scales as
    # (length / time)^2, angular momentum as length^2 / time. Square in float64:
    # the float32 archive values of far-flung orbits overflow when squared.
    abs_l = np.stack([c(f"L{a}_abs_mean").astype(np.float64) for a in "xyz"], axis=1)
    quantities = {
        "energy": c("energy0") * (length_unit / results.time_unit) ** 2,
        "angular_momentum": np.sqrt(np.sum(abs_l**2, axis=1))
        * length_unit**2
        / results.time_unit,
    }
    if pattern is not None:
        quantities["pattern_speed_ratio"] = pattern_speed_ratio

    classification = OrbitClassification(
        labels=labels,
        circulation=circ,
        tube_axis=tube_axis,
        planarity=planarity,
        resonance=res_vec,
        resonance_order=res_ord,
        radius=radius,
        radius_orbit_averaged=radius_orbit_averaged,
        ids=results.ids,
        fundamentals=results.fundamentals,
        quantities=quantities,
        shape=shape,
        boxlet_type=boxlet_type,
        boxlet_plane=boxlet_plane,
        boxlet_elongation=boxlet_elongation,
        boxlet_closed=boxlet_closed,
        second_resonance=sec_vec,
        sense=sense,
        orientation=results.orientation,
        pattern_resonance=pattern_resonance,
        pattern_frequency_ratio=pattern_ratio,
        regularity=regularity,
        unresolved=unresolved,
        pattern_speed_ratio=pattern_speed_ratio,
        corotation_band=tuple(corotation_band),
        family=family,
        family_names=family_names,
        inner_outer_ratio=float(inner_outer_ratio),
        inner_outer_adaptive=adaptive,
    )
    bad = {name: n for name, n in classification.inconsistencies().items() if n}
    if bad:
        logger.warning(f"Inconsistent label combinations: {bad}")
    return classification


def _rotating_frame(results) -> Optional[int]:
    """The pattern-speed axis when the frame-aware tests can run.

    Parameters
    ----------
    results : lanfear.OrbitResults
        The orbits being classified.

    Returns
    -------
    axis : int or None
        The principal axis the figure rotates about (0=x, 1=y, 2=z), or None
        for a static potential. Also None, with a warning, when the pattern
        speed is too far from a principal axis for the exact frame conversion.
    """
    try:
        pattern = results.pattern_axis()
    except ValueError as exc:
        logger.warning(
            f"{exc}; the rotating-figure rosette correction and pattern labels are off"
        )
        return None
    if pattern is None:
        return None
    return pattern[0]
