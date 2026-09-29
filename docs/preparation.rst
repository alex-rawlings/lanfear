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

With two BHs (an SMBH binary), the default ``centre="bh"`` puts the binary's
centre of mass at the origin. Keep it there if the binary matters: orbit
pericentres (``r_peri``), and hence the binary-interacting flag, are measured
from the origin, and a warning is logged if the binary's centre of mass is more
than one semimajor axis away from it (see "SMBH binaries" in :doc:`usage`).

``align()`` then rotates the field's principal axes onto x/y/z, using only the
most bound half of the field particles by default (``bound_fraction=0.5``)
rather than all of them. Boundedness is ranked by an approximate specific energy
from a fast (O(N log N)) spherically-averaged potential estimate, so a diffuse,
often asymmetric envelope or tidal debris does not bias the shape. Pass
``ps.align(bound_fraction=1.0)`` to use every field particle instead, as in
earlier versions.

Figure rotation
---------------

A non-axisymmetric figure (a bar, a tumbling triaxial remnant) may rotate as a
whole at a **pattern speed** ``Omega``. ``prepare()`` estimates it by default
and stores it on ``ps.pattern_speed``, a (3,) angular-velocity vector in the
aligned frame, in physical units (velocity unit / length unit, e.g. km/s/kpc).
Every potential built from the system inherits it (``pot.pattern_speed``), and
rotates rigidly at that rate during orbit integration:

.. code-block:: python

   ps.prepare()                          # pattern_speed="estimate" (default)
   ps.prepare(pattern_speed="none")      # static figure
   ps.prepare(pattern_speed=12.0)        # about the aligned short axis (z)
   ps.prepare(pattern_speed=[0, 0, 12])  # full angular-velocity vector

   est = ps.estimate_pattern_speed()     # the estimate and its diagnostics
   est["pattern_speed"], est["uncertainty"], est["significant"]

   pot = lf.Potential.from_particles(ps, n_max=18, l_max=7)
   pot.pattern_speed                     # inherited; may be reassigned, e.g.
   pot.pattern_speed = "none"            # integrate in a static potential

**Estimate.** ``estimate_pattern_speed()`` works from a single snapshot. It
uses the shape tensor ``T = sum m x x^T / |x|`` of the most bound half of the
field (the particles ``align()`` uses). The particle velocities give the
tensor's exact instantaneous rate of change ``dT/dt``, and a rigidly rotating
figure has ``dT/dt = [Omega x, T]``. The ``1/|x|`` weighting makes each
particle's contribution to ``dT/dt`` its mass times a velocity, which is
bounded everywhere. With ``align()``'s reduced tensor (``1/|x|^2``) the
contributions grow as ``v/|x|`` towards the centre, and a few central particles
would dominate the estimate and its noise. In the principal frame (eigenvalues ``lambda_i``)
each component follows as ``Omega_k = (dT/dt)_ij / (lambda_i - lambda_j)``, for
``(i, j, k)`` cyclic. This is the three-dimensional form of the m = 2 moment
method of Dehnen, Semczuk & Schönrich (2023).

- Ordered streaming in a figure that does not tumble leaves the density, and
  so ``T``, unchanged, and gives no signal.
- Each component has a Poisson uncertainty from the particle-to-particle
  scatter. A component below ``significance`` (default 3) times its
  uncertainty is set to zero.
- Rotation about an axis of symmetry (``lambda_i`` close to ``lambda_j``) is
  undefined, so it is always dropped. A spherical, axisymmetric or
  non-tumbling system therefore ends up with a zero pattern speed and a static
  potential.
- The estimate assumes steady, rigid rotation. A figure that is still changing
  shape (e.g. a young merger remnant) biases it, so check it against
  consecutive snapshots where possible.

**Integration.** Orbits are always integrated in the inertial frame, in the
rotating potential ``Phi(R(t)^T x)``, so a zero pattern speed runs the very
same code as a static potential. Every orbit quantity is measured in the
**co-rotating frame** of the figure, where the potential is static and a
regular orbit is a steady 3-torus. This covers the summary columns, the
fundamental frequencies and ``ParticleTrajectory``:

- Positions are the body-frame ones; velocities are ``dx_b/dt``.
- Angular momenta are ``x_b x dx_b/dt``.
- Resonances are resonances with the pattern.
- The ``energy0``/``energy_mean``/``energy_drift`` columns and
  ``ParticleTrajectory.energy`` hold the **Jacobi integral**
  ``E_J = E - Omega . L``, which a rotating potential conserves. It reduces to
  the energy for a static one.

The classification thresholds were chosen for static potentials, so
treat the families of a strongly rotating figure (in particular near
corotation) with care. Black holes are part of the figure. A combined binary at
the centre is unaffected, but an off-centre black hole co-rotates rigidly with
the figure (a warning is logged).

**Heuristic check.** If the pattern speed ends up zero (``"none"``, or no
significant rotation measured), ``prepare()`` runs the kinematic heuristic
``ps.detect_figure_rotation()``. It logs a ``WARNING`` if the figure
nevertheless looks like it is tumbling: a non-axisymmetric figure with
significant ordered rotation about its short axis. The method returns the axis
ratios, the rotation measure ``|v_rot|/sigma`` and the short axis. Skip it with
``ps.prepare(check_figure_rotation=False)``.

Units
-----

The SCF core works in Hernquist-Ostriker units: ``G = M_field = scale_radius =
1``. The scale radius is estimated from the field half-mass radius as
``r_half / (1 + sqrt(2))`` (exact for a Hernquist profile). The Python layer
converts physical coordinates to/from HO units; black-hole masses and positions
are supplied in physical units and normalised internally.

``MultiComponentPotential`` fits each species its own ``Potential``/
``DiscPotential`` (each with its own scale radius and field mass) and then
superposes them in one shared HO unit system -- by default the Hernquist-style
scale radius of every species' field particles combined, so a single-species
system reduces exactly to that component's own result. See the class
docstring for the full unit-reconciliation derivation; pass ``length_unit=``
to override the default.
