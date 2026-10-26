Classification
===============

``res.classify()`` (:func:`lanfear.classify.classify_orbits`) gives every orbit
several independent labels, each answering one question, and combines them into
one named *family*. This page summarises the scheme. :doc:`usage` shows the
calls, :doc:`tuning` how the thresholds were calibrated, and the API reference
below gives every rule in full.

The labels
----------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Attribute
     - Values
     - What it answers
   * - ``cls.shape`` (``OrbitShape``)
     - ``box``, ``short_axis_tube``, ``inner_long_axis_tube``,
       ``outer_long_axis_tube``, ``intermediate_axis_tube``, ``rosette``,
       ``unclassified``
     - What the orbit looks like over the window: which axis it circulates
       about, if any. Assigned before the irregular override, so a chaotic
       orbit keeps a shape.
   * - ``cls.boxlet_type`` (``BoxletType``)
     - ``banana`` (1:2), ``fish`` (2:3), ``pretzel`` (3:4), ``planar``,
       ``spatial``, ``none``
     - Whether a box is resonant, and which resonance.
   * - ``cls.sense`` (``OrbitSense``)
     - ``prograde``, ``retrograde``, ``locked``, ``librating``, ``none``
     - Which way a loop goes round its axis (see `Axes and sense`_); in a
       rotating figure, whether an orbit is fixed in the inertial frame or
       librates without going round the pattern-speed axis.
   * - ``cls.pattern_resonance`` (``PatternResonance``)
     - ``corotation``, ``x1``, ``x2``, ``x4``, ``olr``, ``ultraharmonic``,
       ``unresolved``, ``none``
     - Rotating figures only: a resonance between the radial and body-frame
       azimuthal frequencies of an orbit going round the pattern-speed axis.
   * - ``cls.regularity`` (``Regularity``)
     - ``regular``, ``irregular`` (more than three base frequencies),
       ``diffusing`` (frequency drift above ``diffusion_threshold``)
     - Whether the orbit is a likely chaotic one.

``cls.labels`` (``OrbitClass``) is the original single label (pi-box, boxlet,
the tubes, rosette, irregular), derived from the shape, boxlet type and
regularity. Every method that takes a ``scheme`` argument works on any label:
``"class"`` (default), ``"family"``, ``"shape"``, ``"sense"``,
``"pattern_resonance"``, ``"regularity"`` or ``"boxlet"``.

Families
--------

``cls.family_name_array`` combines the labels with an ordered rule table,
:data:`lanfear.classify.FAMILY_TABLE`. The first rule that matches an orbit
names its family, and an orbit no rule matches is ``"other"``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Family
     - Rule
   * - ``unclassified``
     - The integration failed.
   * - ``irregular``
     - Regularity ``irregular`` or ``diffusing``, whatever the shape.
   * - ``pattern_locked``
     - Sense ``locked``: barely moves in the inertial frame.
   * - ``corotation_trapped``
     - Librating at corotation, i.e. around the Lagrange points L4/L5.
   * - ``x1``, ``x2``, ``x4``, ``olr``, ``ultraharmonic``
     - The pattern resonance of that name.
   * - ``banana``, ``fish``, ``pretzel``, ``planar_boxlet``,
       ``spatial_boxlet``
     - A resonant box of that type.
   * - ``librating``
     - Avoids the pattern-speed axis without going round it.
   * - ``pibox``
     - Any other box.
   * - ``prograde_<shape>``, ``retrograde_<shape>``
     - A tube or rosette with that sense, e.g.
       ``prograde_short_axis_tube``.
   * - ``<shape>``
     - A tube or rosette without a sense.

Families that only exist in rotating figures come before the boxes and tubes,
so an x1 orbit is ``x1`` whatever its shape. A custom table adds a family
without new classification logic: prepend a :class:`lanfear.FamilyRule` to
``lf.FAMILY_TABLE`` and pass it as ``res.classify(family_table=...)`` (example
in :doc:`usage`).

Boxlet names
------------

``cls.boxlet_names`` names each boxlet by its resonance ``n . w = 0`` among the
axis fundamentals:

* a named two-axis boxlet is ``<type>_<plane>``, e.g. ``banana_xz`` (the plane
  of the two resonant axes);
* any other two-axis boxlet is ``<p>:<q>_<plane>``, its lower:higher in-plane
  frequency ratio, e.g. ``1:3_xy``;
* a spatial (three-axis) boxlet is ``resonant_<nx>_<ny>_<nz>``, e.g.
  ``resonant_1_-2_1``.

``cls.boxlet_plane`` is the axis normal to a two-axis resonance and
``cls.boxlet_elongation`` the axis the boxlet is elongated along.
``cls.boxlet_closed`` marks boxes with two independent resonances (periodic
orbits rather than thin ones). In a rotating figure, a box whose two in-plane
fundamentals are one shared line (within ``shared_line_tol`` frequency
resolutions) is not a 1:1 boxlet: the shared line is the box's precession.

Axes and sense
--------------

:meth:`lanfear.ParticleSystem.prepare` aligns the frame with the principal
axes (x long, y intermediate, z short) and :meth:`~lanfear.ParticleSystem.orient_axes`
fixes their signs, so that "prograde" means the same thing in every run:

* **+z** points along the pattern speed if the figure rotates, and otherwise
  along the net spin ``L_z``;
* **+x** points along the net ``L_x``.

A loop is prograde if it goes round its axis in the positive sense, from the
sign of its azimuthal frequency. About the pattern-speed axis of a rotating
figure that frequency is taken in the inertial frame. If a net ``L_z`` or
``L_x`` is consistent with zero, ``prepare()`` warns that the sign of that
axis, and so prograde/retrograde about it, is arbitrary.
``cls.orientation`` records which reference the results used.

In a rotating figure every quantity is measured in the frame co-rotating with
the figure, where the potential is static. Two rules differ from a static
figure: a loop about the pattern-speed axis also needs a hole about it
(``hole_thresh``), because a box precessing in a tumbling figure winds about
that axis while still passing close to the centre; and rosettes are tested on
the inertial azimuthal frequency.

Reliability flags
-----------------

These do not change any label; they mark orbits whose label is less certain.

* ``cls.unresolved``: one of the orbit's labels depends on a frequency that
  completes fewer than ``resolution_cycles`` (3) cycles in the window.
* ``cls.near_corotation``: in a rotating figure, the orbit's own pattern
  speed ratio ``epsilon = |Omega_p| / |Omega_phi|`` is in ``corotation_band``
  (by default ``epsilon >= 0.5``, at or beyond corotation). On the tumbling
  perfect ellipsoids, labels there are right 45--90% of the time (worst
  beyond corotation), against 95% for the unflagged orbits.
* ``res.runaway``: orbits that leave the system in the window are left out of
  the classification by default (``drop_runaways``).
* ``res.binary_interacting``: orbits whose pericentre reaches a central SMBH
  binary; leave them out with ``drop_binary_interacting=True``.
* ``cls.inconsistencies()``: counts of impossible label combinations (e.g. a
  sense on a box), all zero for a sound classification.

For a probability over the labels rather than flags, see
:doc:`probabilistic_classification`.

How well it works
-----------------

On equilibrium perfect ellipsoids, whose orbit families are known exactly,
the families agree with the true ones for 91--96% of orbits on four static
shapes from prolate to near-oblate. In figures tumbling about their short axis
with corotation at 16 and 8 half-mass radii (slow tumbling, as in
ellipticals) they agree for 94% and 92% (96% for the orbits that are not
chaotic). At 4 half-mass radii the agreement falls to 84%. :doc:`tuning` gives
the calibrated thresholds and the details.

API reference
-------------

.. automodule:: lanfear.classify
   :members:
   :show-inheritance:

Pattern-speed sensitivity
-------------------------

.. automodule:: lanfear.sensitivity
   :members:
   :show-inheritance:
