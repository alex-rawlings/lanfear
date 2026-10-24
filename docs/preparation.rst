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

``align()`` fixes which axis is which, but not their signs: +z could point
either way along the short axis, so "prograde" would change meaning from run to
run. The last step of ``prepare()``, ``orient_axes()``, fixes them with half
turns (the frame stays right-handed):

- +z points along the pattern speed if the figure rotates (``reference="pattern"``),
  otherwise along the net angular momentum of the field (``"spin"``);
- +x points along the net ``L_x`` (any minor-axis rotation).

``"auto"`` (the default) picks ``"pattern"`` when the pattern speed has a z
component. The net angular momenta use the same smooth window as ``align()``.
If a net ``L_z`` (spin reference) or ``L_x`` is below 3 times its Poisson
uncertainty, a warning says the sign of that axis is arbitrary. The choice is
recorded as ``ps.orientation`` and carried into ``res.orientation``.
A pattern speed given to ``prepare()`` refers to the frame before this step and
is rotated with the particles, so its components can change sign.

.. code-block:: python

   ps.prepare()                    # orient="auto"
   ps.prepare(orient="spin")       # +z along the net spin even if rotating
   ps.prepare(orient=None)         # leave the axis signs arbitrary
   ps.orientation                  # "pattern", "spin" or None

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
  uncertainty is set to zero. ``prepare()`` keeps the uncertainty of each
  component (in the aligned frame) as ``ps.pattern_speed_sigma``, also when
  every component was dropped, so it bounds an undetected rotation. The
  potentials inherit it as ``pot.pattern_speed_sigma`` (cleared if the pattern
  speed is reassigned) and the orbit results as ``res.pattern_speed_sigma``. It
  is ``None`` when the pattern speed was given rather than estimated.
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
- Some quantities convert exactly to the inertial frame. About the axis of
  the pattern speed (e.g. z), the planar spectrum ``res.planar_lines[:, 2]``
  (signal ``x_b + i y_b``) is the inertial one shifted by ``-|Omega|``, and the
  unwrapped azimuth ``phi_z_net`` is the inertial one minus
  ``|Omega| t_total``. The radial frequency (``radial_frequency``) is the same
  in both frames. ``res.azimuthal_frequency("inertial")`` and
  ``res.azimuth_net("inertial")`` apply these conversions (about the pattern
  axis only; ``res.pattern_axis()`` rejects a pattern speed more than 5 degrees
  from every principal axis).
- ``res.pattern_speed_ratio()`` gives each circulating orbit's own
  ``epsilon = |Omega| / |Omega_phi|``, with ``Omega_phi`` its inertial
  azimuthal frequency, and ``res.pattern_locked(tol=1e-3)`` flags orbits that
  are fixed in the inertial frame (``|Omega_phi| < tol |Omega|``). On rotating
  test models the locked and unlocked orbits are separated by a clear gap at
  about ``1e-3``.
- A box orbit in a tumbling figure precesses steadily, so over a long window
  it *winds* about the rotation axis (``res.winding_circulation()`` close to
  1) although it passes the centre on every oscillation; its precession is slow
  compared with its radial motion and it has no hole
  (``res.hole_fraction()``).
- The ``energy0``/``energy_mean``/``energy_drift`` columns and
  ``ParticleTrajectory.energy`` hold the **Jacobi integral**
  ``E_J = E - Omega . L``, which a rotating potential conserves. It reduces to
  the energy for a static one.

**Classification of a rotating figure.** ``classify()`` applies three
rotating-figure rules (none of them touches a static potential):

- **Rescued loops.** Near corotation the frame slows a prograde tube's
  circulation until its angular momentum about the rotation axis changes
  sign, so the ``circ_a > circ_thresh`` test misses it. An orbit is rescued as
  a loop about the rotation axis when it winds steadily about it
  (``winding_thresh``, default 0.9), at a rate comparable with its radial
  motion (``|Omega_phi| / omega_R > speed_thresh``, default 0.3: a box
  precessing in the tumbling figure winds too, but slowly), and both
  frequencies complete at least ``rescue_cycles`` (3) cycles. ``cls.rescued``
  marks these orbits; ``classify(rescue=False)`` turns the rescue off.
- **Rosettes** compare the inertial azimuthal frequency of a loop about the
  rotation axis with its out-of-plane frequency, because the frame rotation
  shifts the in-plane body-frame frequencies.
- **Pattern-locked orbits** (``res.pattern_locked()``) are never rosettes
  (``classify(pattern_lock_tol=None)`` disables the check).

On rotating test models the rescue keeps 86–95% of the static short-axis tubes
as loops (78–91% without it), but near corotation it also turns a growing
fraction of static boxes into loops (up to ~60–75% at a corotation radius of
1–2 half-mass radii, against 27–51% without it). Whether those orbits really
are loops in the rotating figure is not yet calibrated, so treat the families
of a strongly rotating figure (in particular near corotation) with care.

**Resonances with the pattern.** Orbits that go round the rotation axis
(loops about it, and librating orbits) get ``cls.pattern_resonance``
(``lf.PatternResonance``) from the radial frequency ``Omega_R`` and the
body-frame azimuthal frequency ``Omega_phi,body = Omega_phi - Omega``, each
condition accepted within ``pattern_tol`` (default 1) frequency resolutions
``2 pi / t_total``:

- ``corotation``: ``Omega_phi,body = 0``. For a librating orbit this means it
  is trapped around L4/L5 (``cls.corotation_trapped``); a loop tagged
  ``corotation`` is simply near-circular at the corotation radius.
- 2:1, ``Omega_R = 2 |Omega_phi,body|``: the closed loops of a bar-like figure
  and the orbits trapped around them. Prograde in the body frame they are
  ``x1`` (elongated along the long axis) or ``x2`` (across it); retrograde
  they are ``x4`` if retrograde in the inertial frame too, otherwise ``olr``
  (outer Lindblad resonance).
- ``ultraharmonic``: 4:1.
- ``unresolved``: ``Omega_R`` or ``Omega_phi,body`` completes fewer than
  ``pattern_cycles`` (3) cycles in the window.

On the rotating test models the ratio ``Omega_R / |Omega_phi,body|`` has a
sharp spike at exactly 2. Placing the same tolerance window at a non-resonant
ratio (2.12) suggests the 2:1 tags are 60--90% genuine (most at faster
rotation), but the 4:1 tag is close to chance level except in fast-rotating
triaxial figures; corotation-trapped (librating) orbits stand far above
chance. The tags are off for a static figure.

**Reliability flags.** Two per-orbit flags mark labels to treat with care:

- ``cls.unresolved``: a frequency one of the orbit's labels depends on (an
  active axis fundamental, the azimuthal frequency giving a loop's sense, or the
  frequencies of the pattern-resonance test) completes fewer than
  ``resolution_cycles`` (3) cycles in the window. On the rotating test models it
  grows from 3--4% (static) to roughly half of all orbits at a corotation radius
  of 1.2 half-mass radii.
- ``cls.near_corotation``: the orbit's own ``epsilon = |Omega| / |Omega_phi|``
  (``cls.pattern_speed_ratio``, from ``res.pattern_speed_ratio()``) lies in
  ``corotation_band`` (default ``(0.5, 2.0)``, provisional). ``epsilon`` is also
  available to ``cls.fractions_by(edges, quantity="pattern_speed_ratio")``.

``cls.regularity`` (``lf.Regularity``) keeps the reason an orbit's family label
is ``irregular``: ``irregular`` (more than three base frequencies) or
``diffusing`` (frequency diffusion above ``diffusion_threshold``).

**Sensitivity to the pattern speed.** The estimated pattern speed has an
uncertainty (``ps.pattern_speed_sigma``, inherited by the potential).
``lf.pattern_speed_sensitivity`` re-integrates a random subsample at
``Omega - n sigma``, ``Omega`` and ``Omega + n sigma`` about the rotation axis
and reports which labels change. When no rotation was detected (pattern speed
set to zero) it tests ``+-n sigma`` about z, i.e. whether an undetected rotation
would matter. It runs under MPI like ``analyse_family`` and restores the
potential's pattern speed afterwards.

.. code-block:: python

   sens = lf.pattern_speed_sensitivity(pot, ps, n_sigma=1.0, subsample=2000)
   sens.flip_fractions     # {"labels": ..., "shape": ..., "sense": ..., "any": ...}
   sens.flipped            # (n,) family label or runaway status changes
   sens.classifications    # the three classifications (minus, nominal, plus)

**Runaways.** Stars whose velocities do not suit the rotating potential
(typically beyond corotation) can leave the system altogether.
``res.runaway`` flags orbits whose maximum radius exceeds ``runaway_factor``
(default 10) times the larger of their initial radius and the system radius
(99.9th percentile of the initial radii). ``analyse_family`` logs a
``WARNING`` with their number. Their labels would be meaningless, so
``classify()`` and the probabilistic classifier leave them out **by default**.
The classification then holds fewer orbits than the results: match the two by
particle ID (``cls.ids``), or classify ``res.drop_runaways()`` so both hold the
same orbits. ``classify(drop_runaways=False)`` keeps every orbit (with a
warning). The flag works for static potentials too, where it catches unbound
stars.

.. code-block:: python

   res.runaway                                 # (N,) bool: r_max > factor * max(r0, system radius)
   res.runaway_factor = 20.0                   # may be changed after integration
   cls = res.classify()                        # runaways excluded (default)
   kept = res.drop_runaways()                  # results aligned with cls
   cls_all = res.classify(drop_runaways=False) # every orbit, one label each

**Integration length.** ``n_periods`` counts each orbit's local circular
period, but near corotation its frequencies in the co-rotating frame, where it
is analysed, tend to zero: its body-frame period
``2 pi / |Omega_c - |Omega||`` (summary column ``body_period``; the worst case,
a prograde circular orbit) can be many inertial periods long, and
``n_periods`` inertial periods would leave the frequency analysis unresolved.
For a rotating figure, ``analyse_family`` therefore lengthens each orbit's
window by the power of two nearest to ``body_period / period``, so that it
spans at least ``n_periods / sqrt(2)`` body-frame periods as well, with
``n_samples`` raised by the same factor (so the sampling interval is
unchanged), up to ``max_body_period_factor`` (default 8; 1 disables it).
Orbits with ``|Omega| / Omega_c`` below ~0.3 are not lengthened. The
``t_total`` column records the actual length, and
``res.body_periods_integrated`` the body-frame periods covered; orbits so close
to corotation that the cap was reached are logged with a ``WARNING``. The cost
grows with the fraction of orbits near corotation.

.. code-block:: python

   res = lf.analyse_family(pot, ps, n_periods=50, max_body_period_factor=8)
   res.body_periods_integrated                 # (N,) body-frame periods covered

Black holes are part of the figure. A combined binary at
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
