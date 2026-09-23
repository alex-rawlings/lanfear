Usage
=====

This page walks through the full pipeline in one script. The central black hole
is handled specially: an SCF basis cannot represent a point mass, so the
potential is expanded over **everything except the BH**, and the BH is
re-attached afterwards as a softened point mass **at its actual position**
(which need not be the origin).

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
   ps.prepare()             # recentre (BH CoM), align (most-bound 50% of field), scale radius, figure-rotation check

   # Spherical-ish systems: Hernquist-Ostriker basis.
   pot = lf.Potential.from_particles(ps, n_max=18, l_max=7)
   # Flattened / disc-like systems: Miyamoto-Nagai disc basis (same interface).
   #   pot = lf.DiscPotential.from_particles(ps, n_radial=10, n_vert=3)
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
   # launched under srun/mpirun, otherwise serial):
   res = lf.analyse_family(pot, ps, family="STAR", n_periods=50, n_lines=4)
   if res is not None:                           # None on non-root MPI ranks
       print(res.column("energy_drift"))         # per-orbit summary columns
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
