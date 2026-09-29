Usage
=====

This page walks through the full pipeline in one script. The central black hole
is handled specially: an SCF basis cannot represent a point mass, so the
potential is expanded over **everything except the BH**, and the BH is
re-attached afterwards as a softened point mass **at its actual position**
(which need not be the origin). A snapshot with two BHs may hold an SMBH
binary; see `SMBH binaries`_ below.

Everything is available on the top-level package (``import lanfear as lf``).
See :doc:`preparation` for centring, alignment and units, :doc:`running` for
MPI, logging and progress output, :doc:`tuning` for choosing the potential
orders, the chaos threshold and the integration length, and
:doc:`probabilistic_classification` for per-orbit posterior probabilities on
top of the classification below.

Quickstart
----------

.. code-block:: python

   import lanfear as lf

   ps = lf.ParticleSystem.from_gadget_hdf5("snapshot.hdf5")
   ps.prepare()             # recentre (BH CoM), align (most-bound 50% of field), scale radius, pattern speed
   #   ps.prepare(pattern_speed="none")   # static figure instead (see "Figure rotation" in the preparation docs)

   # Spherical-ish systems: Hernquist-Ostriker basis.
   pot = lf.Potential.from_particles(ps, n_max=18, l_max=7)
   # Flattened / disc-like systems: Miyamoto-Nagai disc basis (same interface).
   #   pot = lf.DiscPotential.from_particles(ps, n_radial=10, n_vert=3)
   #
   # A system built from physically distinct species -- e.g. a stellar disc
   # embedded in a dark-matter halo -- gets an independent fit PER SPECIES,
   # superposed into one potential (exact, since Poisson's equation is linear).
   # Every species present in the snapshot needs an explicit spec; there is no
   # default potential type, since guessing wrong would silently fit badly:
   #   pot = lf.MultiComponentPotential.from_particles(
   #       ps,
   #       components={
   #           "STAR": lf.disc_component(n_radial=8, n_vert=3),
   #           "DM": lf.scf_component(n_max=18, l_max=7),
   #       },
   #   )
   #   pot.validate_component("DM")   # goodness-of-fit of one species alone
   result = pot.validate()                       # analytic potential vs direct sum
   print(result)                                 # median / p90 / worst rel. error
   assert result.passed(tolerance=0.02)

   # Unsure how to pick n_max / l_max? Sweep them and watch the error converge:
   sweep = lf.Potential.truncation_convergence(
       ps, n_max_values=[2, 6, 12, 18], l_max_values=[0, 2, 4, 6]
   )
   sweep.plot()                                  # median error vs n_max and vs l_max

   # Evaluate (HO units) at physical coordinates:
   phi = pot.potential([[1.0, 0.0, 0.0]])
   acc = pot.acceleration([[1.0, 0.0, 0.0]])

   # The potential is built from ALL particles, but integration can be restricted
   # to a spatial region: `radius_mask` composes with `select` like `species_mask`.
   inner = ps.select(ps.radius_mask(r_max=10.0))         # subset within r
   res = lf.analyse_family(pot, inner, family="STAR")    # only inner stars integrated
   #   combine masks: ps.select(ps.species_mask("STAR") & ps.radius_mask(10.0))

   # Integrate every star for 50 orbital periods and frequency-analyse it
   # (fundamentals + spectral lines per axis, in one pass; MPI-distributed if
   # launched under srun/mpirun, otherwise serial). The potential rotates at
   # pot.pattern_speed; everything is measured in the co-rotating frame:
   res = lf.analyse_family(pot, ps, family="STAR", n_periods=50, n_lines=4)
   if res is not None:                           # None on non-root MPI ranks
       res.pattern_speed                         # (3,) figure angular velocity used
       print(res.column("energy_drift"))         # per-orbit summary columns (E_J drift if rotating)
       print(res.column("r_peri"))               # pericentre (HO), resolved by every integrator step
       print(lf.SUMMARY_COLUMNS)                 # available quantities
       good = res.ok                             # status == 0

       res.fundamentals        # (N, 3) signed fundamental frequency per axis (HO)
       res.lines               # (N, 3, n_lines, 2) leading (freq, amp) per axis
       res.frequency_ratios    # (N, 2) |w_x|/|w_z|, |w_y|/|w_z|

       # Laskar frequency diffusion |w2 - w1|/|w1| between the first and second
       # halves of each integration: ~0 for regular orbits, large for chaotic ones.
       res.diffusion           # (N, 3) per axis (NaN where not measurable)
       res.diffusion_rate()    # (N,) max over the axes that actually oscillate

       # Integration is expensive -- save the results and reload later without
       # re-integrating (a compact .npz holding everything OrbitResults needs):
       res.save("orbits.npz")
       res = lf.OrbitResults.load("orbits.npz")   # resume classification/plotting

       # Classify into orbit families (pi-box / tube / rosette / boxlet /
       # irregular ...). An orbit whose spectrum needs > 3 base frequencies is
       # labelled `irregular` (Frigo et al. 2021) -- a likely-chaotic candidate.
       # Orbits whose frequencies drift by more than `diffusion_threshold` (default
       # 0.1) between the two integration halves are also labelled `irregular`.
       # Check np.log10(res.diffusion_rate()) first: lower the cut (e.g. 1e-2) to
       # catch weakly chaotic orbits, or pass None to disable it.
       #   cls = res.classify(diffusion_threshold=1e-2)
       cls = res.classify()
       cls.labels              # (N,) lf.OrbitClass values
       cls.names               # (N,) family name strings
       cls.counts()            # {family_name: count}
       z_tubes = cls.mask(lf.OrbitClass.SHORT_AXIS_TUBE)
       chaotic = cls.mask(lf.OrbitClass.IRREGULAR)
       z_tube_ids = cls.get_class_ids(lf.OrbitClass.SHORT_AXIS_TUBE)  # particle IDs

       # Condense the subclasses into the box / tube dichotomy:
       fam = cls.condense_families()
       fam.counts()            # {'box': ..., 'tube': ..., 'unclassified': ...}
       tubes = fam.mask(lf.OrbitFamily.TUBE)

       # Radial profile of the orbit-class mix (returns a matplotlib Axes):
       import numpy as np
       ax = cls.plot_class_fractions(np.linspace(0, 20, 11))   # fraction within each bin
       ax.figure.savefig("class_fractions.png")
       #   per_bin=False normalises to the total orbit count instead.

       # Bin on energy or angular momentum instead of radius, and shade a
       # bootstrap confidence band (resampling orbits) on each curve:
       ax = cls.plot_class_fractions(
           np.linspace(-4, 0, 9), quantity="energy", n_bootstrap=200, seed=0
       )

       # The numbers behind the plot (fractions, counts, lower/upper bounds):
       frac = cls.fractions_by(np.linspace(0, 20, 11), quantity="radius", n_bootstrap=200)
       frac.fractions, frac.lower, frac.upper   # each (n_classes, n_bins)

       # Bar chart of the orbit count per class:
       ax = cls.plot_class_histograms()
       ax.figure.savefig("class_histogram.png")

       # Frequency map: scatter of (|w_x|/|w_z|, |w_y|/|w_z|) coloured by class,
       # where regular families cluster and resonances trace straight lines:
       ax = cls.plot_frequency_map()
       ax.figure.savefig("frequency_map.png")

       # Every orbit a single hard label -- but some sit right at one of the
       # thresholds above. For a posterior probability over every class
       # instead, see :doc:`probabilistic_classification`:
       #   prob = res.classify_probabilistic()

SMBH binaries
-------------

When a snapshot holds exactly two BH particles, ``from_particles`` treats them as
a (possible) binary. It computes the pair's Keplerian semimajor axis ``a`` and
eccentricity from their relative position and velocity, and its influence
radius ``r_infl`` (the radius, about the binary's centre of mass, enclosing
twice the binary mass in field particles). These are stored on
``pot.binary``.

How the binary is attached is set by ``binary_treatment``:

- ``"auto"`` (default): a bound binary with ``a < r_infl`` is attached as **one
  point mass** ``M_1 + M_2`` at its centre of mass; a wider (or unbound) pair
  keeps **two separate softened masses** (each softened with ``bh_softening``).
- ``"point"``: always combine the pair.
- ``"separate"``: always keep two masses (the behaviour before binaries were
  recognised).

A bound binary orbits much faster than any field orbit outside it, so to those
orbits it acts like a single point mass. Freezing the two BHs at their snapshot
positions instead imposes a static two-centre field that no star experiences,
and makes the result depend on the binary's orbital phase at the snapshot.

A combined binary is softened with a spline softening length **equal to its
semimajor axis** ``a`` (its separation, if an unbound pair is forced with
``"point"``), overriding ``bh_softening``. The spline kernel is exactly
Newtonian beyond its softening length, so the field is exactly Keplerian outside
the binary, where a point mass is valid, and smooth and bounded inside it,
where no static model is right anyway. This also ties the resolution limit to
a physical scale instead of an arbitrary one, and avoids the stiff, deep well a
tiny softening would put at the centre.

An orbit whose pericentre reaches the binary cannot be represented by *any*
static potential: in reality it is scattered (slingshot) by the binary. So
``analyse_family`` flags every orbit whose pericentre ``r_peri`` (the minimum
radius over every integrator step, which resolves fast pericentre passages the
output samples can miss) comes within ``binary_interaction_factor`` (default 1)
binary semimajor axes of the centre. Pericentres are measured from the origin,
so recentre on the BHs (``prepare(centre="bh")``, the default) first. The
flagged orbits are kept, and can be dropped from any downstream step:

.. code-block:: python

   pot = lf.Potential.from_particles(ps, n_max=18, l_max=7)     # binary_treatment="auto"
   pot.binary                          # BinaryProperties (a, e, r_infl, ...), or None
   pot.binary.compact                  # True -> attached as one point mass

   res = lf.analyse_family(pot, ps, family="STAR", binary_interaction_factor=1.0)
   res.binary_interacting              # (N,) bool: r_peri < factor * a
   res.column("r_peri")                # pericentre per orbit (HO units)

   cls = res.classify(drop_binary_interacting=True)              # flagged orbits excluded
   prob = res.classify_probabilistic(drop_binary_interacting=True)
   kept = res.drop_binary_interacting()                          # an OrbitResults without them

   # The factor can be changed after integration; the flag is recomputed.
   res.binary_interaction_factor = 3.0

The binary's semimajor axis and the interaction factor are saved with the
results, so a reloaded ``OrbitResults`` keeps the flag. Archives written before
``r_peri`` existed fall back on the sampled ``r_min`` (with a warning).

Comparing snapshots
-------------------

Compare two snapshots (e.g. before/after a perturbation) particle-by-particle,
matched by particle ID. Particles in only one snapshot are dropped, and it is up
to you which snapshot is the earlier one:

.. code-block:: python

   before = res_early.classify()
   after = res_late.classify()

   cmp = before.compare(after)          # `before` is the "before" state by convention
   cmp.n_matched                        # particles present in both
   cmp.fraction_changed                 # fraction that switched family at any stage
   cmp.changed                          # (M,) bool, per matched particle (by cmp.ids)
   rows, cols, matrix = cmp.transition_matrix()   # counts of before-class -> after-class

   # Sankey diagram of the family flow from `this` (before) to `other` (after):
   ax = cmp.plot_sankey()
   ax.figure.savefig("family_flow.png")

``compare()`` also accepts an iterable of classifications to compare more than
two snapshots in sequence (``self`` followed by each item, in order), producing
one stage per classification, e.g. ``before.compare([mid, after])``.
``transition_matrix(stage=i)`` then indexes the ``i``-th consecutive pair, and
``plot_sankey()`` draws every stage as one multi-column diagram.

Both classifications must use the same class scheme: condense both with
``condense_families()`` first, or compare two full classifications.

Plotting a single trajectory
----------------------------

Storing every particle's full phase-space trajectory "just in case" is wasteful
when only a handful are ever inspected in detail, so ``ParticleTrajectory``
re-integrates just the one orbit you ask for (rather than reading it back out of
a batch ``analyse_family``/``analyse_states`` run) and plots it:

.. code-block:: python

   traj = lf.ParticleTrajectory.from_particles(pot, ps, particle_id=12345, n_periods=10)
   # or, without a ParticleSystem, from a physical position/velocity directly:
   #   traj = lf.ParticleTrajectory.integrate(pot, pos_phys, vel_phys, n_periods=10)

   axes = traj.plot()              # x-y, x-z, y-z projections, coloured by time (BuPu)
   axes[0].figure.savefig("trajectory.png")

The trajectory also keeps the sampled velocities (``traj.vel``) and the
specific orbital energy ``0.5 |v|^2 + Phi`` in physical units (``traj.energy``).
``plot_energy()`` plots it against time, as a check on energy conservation:
by default it shows the relative drift ``(E - E0) / |E0|``, whose maximum
magnitude is the ``energy_drift`` summary column; pass ``relative=False`` for
the energy itself.

If the potential rotates (a non-zero ``pot.pattern_speed``), the trajectory is
recorded in the co-rotating frame of the figure (``traj.pos``, and
``traj.vel = d traj.pos / dt``). ``traj.energy`` is then the conserved Jacobi
integral ``0.5 |vel|^2 + Phi - 0.5 |Omega x pos|^2``. See "Figure rotation"
in :doc:`preparation`.

.. code-block:: python

   ax = traj.plot_energy()         # relative energy drift against time
   ax.figure.savefig("trajectory_energy.png")
