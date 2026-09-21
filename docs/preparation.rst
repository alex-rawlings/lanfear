Preparing the particle system
=============================

Centring and alignment
----------------------

``prepare()`` recentres on the black-hole centre of mass by default
(``centre="bh"``, both position and velocity). Systems with no BH particles must
pass an explicit alternative, typically ``centre="shrinking_sphere"`` (robust to
a distant, mass-biasing stream or outlier population; see
``shrinking_sphere_centre()``), or ``centre="field"`` / a species label for a
plain mass-weighted centre. ``recentre()`` raises ``ValueError`` rather than
silently falling back if the requested selection matches no particles.

``align()`` then rotates the field's principal axes onto x/y/z, using only the
most bound half of the field particles by default (``bound_fraction=0.5``)
rather than all of them. Boundedness is ranked by an approximate specific energy
from a fast (O(N log N)) spherically-averaged potential estimate, so a diffuse,
often asymmetric envelope or tidal debris does not bias the shape. Pass
``ps.align(bound_fraction=1.0)`` to use every field particle instead, as in
earlier versions.

Figure rotation
---------------

``prepare()`` also checks for **figure rotation** (a tumbling, non-axisymmetric
figure) and logs a ``WARNING`` if it is detected: the classifier integrates
orbits in a *static* potential, so a tumbling figure would produce erroneous
orbit families. Inspect the diagnostics directly with
``ps.detect_figure_rotation()`` (returns the axis ratios, the rotation measure
``|v_rot|/sigma``, and the short axis), or skip the check with
``ps.prepare(check_figure_rotation=False)``.

Single-snapshot detection is necessarily heuristic: it flags a non-axisymmetric
figure with significant ordered rotation about its short axis. Confirm the actual
pattern speed from consecutive snapshots before discarding a classification.

Units
-----

The SCF core works in Hernquist-Ostriker units: ``G = M_field = scale_radius =
1``. The scale radius is estimated from the field half-mass radius as
``r_half / (1 + sqrt(2))`` (exact for a Hernquist profile). The Python layer
converts physical coordinates to/from HO units; black-hole masses and positions
are supplied in physical units and normalised internally.
