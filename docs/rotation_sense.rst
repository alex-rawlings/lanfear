Rotation sense
==============

The loop ``sense`` of the classification (``cls.sense``, see :doc:`classification`)
says which way each loop goes round **its own** circulation axis, with the axis
signs fixed by ``orient_axes()``. It is the default and is unchanged.

This page covers a separate, opt-in label: whether an orbit **co-rotates or
counter-rotates with a chosen reference**, such as the whole field, the stars,
the dark matter, or the stars within some radius. It applies to every orbit,
boxes included, and works when components are misaligned, e.g. a halo spinning
about a different axis from the disc.

References
----------

A reference is the direction ``n`` of the net angular momentum
``sum m x x v`` of a set of particles, measured from the prepared snapshot:

.. code-block:: python

   ps.prepare()
   pot = lf.MultiComponentPotential.from_particles(ps, components={...})

   ref = ps.reference_spin()                                 # whole field ("field")
   ref = ps.reference_spin(family="DM")                      # one family ("DM")
   ref = ps.reference_spin(family="STAR", radius=5.0, potential=pot)  # within 5
   ref = ps.reference_spin(family="STAR", radius=(2.0, 8.0), potential=pot)  # shell
   ref = ps.reference_spin(family="STAR", radius=5.0, geometry="cylindrical", z_max=1.0)
   ref = ps.reference_spin(mask=my_mask, name="my_selection")

   ref.axis, ref.angle_uncertainty, ref.significant, ref.n_particles

Radii are in the particles' physical length units. ``family=None`` selects every
field (non-BH) particle.

**Aperture shape.** With a radius, ``geometry`` sets the shape:

- ``"spherical"``: ``inner <= r < outer``.
- ``"cylindrical"``: ``inner <= R < outer`` and ``|z| < z_max``, about the
  selection's **own** short axis. That axis is found by iterating: start from
  the spherical aperture, take the short axis of ``sum m x x^T``, re-cut the
  cylinder about it, and repeat until it settles. A disc tilted against the
  aligned frame is therefore still cut correctly. It is stored as
  ``ref.cylinder_axis``. ``z_max`` defaults to 3 times the median ``|z|`` of the
  selection inside ``R``, and is stored as ``ref.z_max``. The default follows
  the whole selection, so a thick component of the same family (a bulge or
  stellar halo in ``"STAR"``) inflates it and pulls that component into the
  cylinder: give a disc-scale ``z_max`` explicitly then.
- ``"auto"`` (default): cylindrical for families fitted with a disc basis in
  ``potential`` (``disc_component`` / ``DiscPotential``), spherical otherwise.
  It is decided from the fit only, never from the particles' shape, and raises
  ``ValueError`` if there is no potential or the selected families mix disc and
  spheroidal fits.

``window="gaussian"`` weights every particle by ``exp(-r^2 / 2 R^2)`` instead
of cutting at ``R`` (spherical, no inner radius).

**Names** are generated from the selection, e.g. ``"field"``, ``"DM"``,
``"STAR_r<5"``, ``"STAR+DM_2<=r<8"``, ``"STAR_R<5_|z|<0.49"``.

**Significance.** Each component of the net spin has a Poisson uncertainty
from the particle-to-particle scatter, which also gives
``ref.angle_uncertainty``. If the spin along ``n`` is below ``significance``
(default 3) times its uncertainty, a warning is logged and
``ref.significant`` is False. Every orbit's sense relative to it is then
``undefined``. This is typical for a dark-matter halo with little net rotation.

**Frame.** A reference records the frame of the particles it was measured from
(``ref.frame``, see "Frame tracking" in :doc:`preparation`). It can only be
used with orbits in the same frame; anything else raises ``ValueError``. This
stops a re-prepared snapshot with flipped axes from silently swapping co and
counter.

``lf.ReferenceSpec(family="STAR", radius=5.0)`` holds the same arguments for
later, and ``spec.resolve(ps, potential=pot)`` builds the reference.
``lf.SpinReference.from_axis(axis, ps.frame, name)`` is a reference along a
given direction instead, e.g. the pattern-speed axis or +z (always
significant).

Per-orbit measures
------------------

The comparison uses the inertial angular-momentum moments ``<L>`` and
``<L L^T>`` of each orbit (the ``Lin_*`` summary columns), so references can be
added after integration without re-running it:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Method
     - Definition
     - Meaning
   * - ``res.circulation_coherence()``
     - ``kappa = |<L>| / sqrt(tr <L L^T>)``, in [0, 1]
     - Whether the orbit has a net circulation at all (about 0 for a box). The
       same in every frame.
   * - ``res.reference_cosine(ref)``
     - ``<L> . n / |<L>|``
     - Which way it circulates relative to the reference.
   * - ``res.reference_sign_consistency(ref)``
     - ``<L> . n / sqrt(n^T <L L^T> n)``
     - Whether ``L . n`` keeps its sign. A diagnostic only: a circular orbit
       tilted 89 degrees from ``n`` still scores 1.

Labels
------

.. code-block:: python

   res.add_reference(ps.reference_spin(family="STAR", radius=5.0, potential=pot))
   res.references                       # stored by name; saved with the results
   sense = res.rotation_sense("STAR_R<5_|z|<0.49")
   names = np.array([lf.ROTATION_SENSE_NAMES[int(s)] for s in sense])

   res.rotation_sense(ref, coherence_min=0.2, perpendicular=0.5)  # the defaults
   res.rotation_sense(ref, perpendicular=None)                     # co/counter only

``res.rotation_sense(ref)`` gives each orbit a ``lf.RotationSense``, in this
order:

- ``undefined``: the reference's spin is not significant (every orbit);
- ``none``: a failed orbit, or coherence below ``coherence_min``;
- ``perpendicular``: ``|cos theta| < perpendicular``, i.e. between 60 and 120
  degrees by default;
- ``co`` (``cos theta > 0``) or ``counter`` (``cos theta < 0``).

**Perpendicular orbits.** For a reference along z, a short-axis tube has
``<L>`` along z and is ``co`` or ``counter``. A long-axis (polar) tube has
``<L>`` along x, about 90 degrees from the reference. It circulates, but not
about the reference, so it is ``perpendicular``. Without this class its label
would come from the sign of a small, noisy ``L_z``. Its sense about its own
axis is still in ``cls.sense``.

**Tumbling figures.** The moments are inertial, so a box orbit in a tumbling
figure, which precesses with the pattern, has a net spin along the pattern
speed and is usually ``co`` relative to it. That is physical, since such
orbits carry the figure's angular momentum, but it means that in a tumbling
figure ``none`` is not a synonym for "box".

In the classification
---------------------

``classify(references=...)`` adds one label scheme per reference. It takes
names of stored references, reference objects, or ``"all"``. Without it,
nothing changes.

.. code-block:: python

   res.add_reference(ps.reference_spin(family="STAR", radius=5.0, potential=pot))
   res.add_reference(ps.reference_spin(family="DM"))
   cls = res.classify(references="all")

   cls.rotation_sense_schemes            # ("rotation_sense:STAR_R<5_|z|<0.49", "rotation_sense:DM")
   cls.rotation_sense_names("DM")        # (N,) "co", "counter", ...
   cls.counts("rotation_sense:DM", particles=ps)           # mass per label
   cls.fractions_by(edges, scheme="rotation_sense:DM")     # versus radius
   cls.crosstab("family", "rotation_sense:DM")             # per family

   # Continuous versions, for binning or your own cuts:
   cls.fractions_by(edges, quantity="circulation_coherence")
   cls.fractions_by([-1, -0.5, 0.5, 1], quantity="reference_cosine:DM")

``coherence_min`` and ``perpendicular`` are passed on from ``classify()``.
``lf.classify_orbits_probabilistic(res, references="all")`` also gives every
reference a probabilistic scheme (see :doc:`probabilistic_classification`).

Calibration
-----------

The defaults were calibrated on the static and tumbling perfect ellipsoids
and checked on synthetic systems with known kinematics (see "Rotation-sense
thresholds" in :doc:`tuning`):

- ``coherence_min = 0.2`` separates boxes from tubes in a static figure with a
  balanced accuracy of 0.987.
- ``perpendicular = 0.5`` (60--120 degrees) puts 98% of short-axis tubes and
  all inner long-axis tubes on the right side for a reference along z.
- ``co``/``counter`` agree with the true sense for 98.7--99.85% of the labelled
  orbits, static and tumbling. On the mocks (a 30% counter-rotating fraction,
  misaligned disc and halo spins, a counter-rotating core) the agreement is at
  least 99.8%.

API reference
-------------

.. automodule:: lanfear.rotation_sense
   :members:
   :show-inheritance:
