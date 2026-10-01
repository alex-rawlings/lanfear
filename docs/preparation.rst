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

``align()`` then rotates the field's principal axes onto x/y/z (longest axis to
x, shortest to z). It diagonalises the shape tensor ``sum m r g(r) u u^T`` of
every field particle (``u = x / r``), weighted by the smooth radial window
``g(r) = exp(-r^2 / 2R^2)``. ``R = window_radius`` defaults to the field
half-mass radius, e.g. ``ps.align(window_radius=5.0)``. The window keeps a
diffuse, often asymmetric envelope or tidal debris from biasing the shape. It
depends on position only, so the axes do not depend on the kinematics.
Earlier versions selected the most bound half of the field by an approximate
energy, which ties the axes to the velocities (and, in a tumbling figure, to
the sense of rotation). This is the same tensor the pattern-speed estimate
uses (see below), so after alignment its principal axes are x, y and z.

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
uses **every** field particle, weighted by a smooth radial window
``g(r) = exp(-r^2 / 2R^2)`` (``R = window_radius``, by default the field
half-mass radius), through the shape tensor ``T = sum m r g(r) u u^T``
(``u = x / r``). The particle velocities give the tensor's exact instantaneous
rate of change ``dT/dt``, and a rigidly rotating figure has
``dT/dt = [Omega x, T]``. In the principal frame (eigenvalues ``lambda_i``)
each component follows as ``Omega_k = (dT/dt)_ij / (lambda_i - lambda_j)``, for
``(i, j, k)`` cyclic. This is the three-dimensional form of the moment method
of Dehnen, Semczuk & Schönrich (2023).

Two choices make the estimate unbiased for a steady figure:

- **No selection by velocity.** Splitting each velocity into ``Omega x x`` and
  the rotating-frame velocity, the estimate is ``Omega`` plus a streaming term.
  That term is the window-weighted divergence of the steady rotating-frame
  flow and averages to zero, but only if particles are not chosen by anything
  that depends on velocity. An earlier version used the most bound half by
  *inertial* energy. The inertial energy depends on velocity and is not
  conserved in a tumbling figure, and the selection favoured particles moving
  against the rotation, which biased the estimate low (by ~10-40% in steady
  tumbling test populations).
- **A smooth window, not a hard edge.** A hard aperture (``r < R``) leaves a
  flux term through its surface, which bar-like streaming does not cancel.
  The ``1/r`` factor in the weight makes each particle's contribution to
  ``dT/dt`` its mass times a velocity, bounded at both small and large radius.

Further properties:

- Ordered streaming in a figure that does not tumble leaves the density, and
  so ``T``, unchanged, and gives no signal.
- Each component has a Poisson uncertainty from the particle-to-particle
  scatter. A component below ``significance`` (default 3) times its
  uncertainty is set to zero.
- Rotation about an axis of symmetry (``lambda_i`` close to ``lambda_j``) is
  undefined, so it is always dropped. A spherical, axisymmetric or
  non-tumbling system therefore ends up with a zero pattern speed and a static
  potential.
- The estimate assumes steady, rigid rotation. When a pattern speed is found,
  it is compared with ``ps.pattern_speed_profile()``, the same estimator in
  overlapping log-normal radial shells (0.25, 0.5, 1 and 2 window radii by
  default). A ``WARNING`` is logged if any shell disagrees by more than
  ``significance`` times its uncertainty: the figure rotates differentially,
  or is still changing shape (e.g. a young merger remnant), so a single
  pattern speed is only an average. The profile is returned as
  ``est["profile"]``.

.. code-block:: python

   est = ps.estimate_pattern_speed(window_radius=5.0)   # explicit window scale
   prof = ps.pattern_speed_profile(radii=[1, 2, 4, 8])  # Omega in radial shells
   prof["radius"], prof["pattern_speed"], prof["uncertainty"]

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
