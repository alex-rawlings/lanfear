Probabilistic classification
=============================

:doc:`usage`'s ``res.classify()`` assigns each orbit a single family from a
handful of hard-thresholded tests (circulation, resonance, frequency
diffusion, ...). Some orbits sit right at one of those thresholds, where a
hard label is not really a well-defined physical statement -- an orbit whose
circulation is 0.89 against a cut of 0.9, or whose diffusion rate is a factor
of 1.2 below the chaos threshold, is genuinely ambiguous, not just noisily
measured.

``res.classify_probabilistic()`` adds an optional layer on top of that: a
posterior probability over every class for each orbit, fitted from the same
cheap, already-computed per-orbit quantities ``classify()`` itself uses. It
does not replace ``classify()`` or change its output -- it is trained *on*
``classify()``'s own labels (restricted to the orbits confidently away from
every threshold it applies) and only adds calibrated uncertainty around the
boundaries that classifier already draws. See :doc:`tuning` for those
thresholds themselves.

Quickstart
----------

Everything below works out of the box on any ``OrbitResults`` that already
has frequency data (i.e. anything from ``analyse_family``/``analyse_states``,
or reloaded with ``OrbitResults.load``) -- no separate training set or
feature engineering is needed:

.. code-block:: python

   import lanfear as lf

   res = lf.analyse_family(pot, ps, family="STAR", n_periods=50, n_lines=4)

   # Fits a Bayesian model self-supervised on classify_orbits()'s own labels
   # (the orbits confidently away from its thresholds), then evaluates it on
   # every orbit -- one call, sensible defaults throughout:
   prob = res.classify_probabilistic()

   prob.labels                 # (N,) MAP lf.OrbitClass value (-1 = unevaluable)
   prob.map_probability        # (N,) posterior probability of that MAP label
   prob.entropy                # (N,) Shannon entropy of the posterior, in bits
   prob.counts()                # {family_name: count}, like OrbitClassification.counts()

   # Bar chart of the full posterior for one particle:
   names, p = prob.probability_bar(id=123456)
   ax = prob.plot_probability_bar(id=123456)
   ax.figure.savefig("particle_123456_posterior.png")

   # Which particles is the model unsure about? (map_probability < 0.8 by default)
   ambiguous = prob.mask_ambiguous()
   confident = prob.mask_confident()
   print(f"{ambiguous.sum()} of {len(ambiguous)} orbits are ambiguously classified")

   # Mean posterior breakdown for a whole subset, e.g. everything currently
   # MAP-labelled boxlet -- a soft view of how "boxlet-like" that population
   # really is:
   names, mean_p = prob.probabilities_for(cls=lf.OrbitClass.BOXLET)
   ax = prob.plot_probability_bar(cls=lf.OrbitClass.BOXLET)

   # Feed the MAP labels straight into the existing OrbitClassification
   # plotting/comparison machinery:
   hard = prob.to_hard()
   ax = hard.plot_class_fractions(np.linspace(0, 20, 11))

Fit once, reuse
----------------

Fitting is a single closed-form pass over a handful of reduced features (a
few seconds even for a large population) and the resulting model is a few KB
-- but the *quality* of the fit depends on the population it is trained on,
so treat fitting as a one-off calibration step, not something to redo on
every run:

.. code-block:: python

   # Fit once on a representative population and save it:
   model = lf.fit_probabilistic_classifier(res)
   model.save("orbit_classifier.npz")

   # Reuse it on later runs, or on other populations from the same potential
   # regime, without refitting:
   model = lf.BayesianOrbitClassifier.load("orbit_classifier.npz")
   prob = res_new.classify_probabilistic(model=model)

Refit when either of these changes:

* **The population regime.** The fit is specific to the feature distributions
  of the population it was trained on (a given potential's triaxiality,
  flattening, ...). A model trained on a strongly triaxial population is not
  guaranteed to transfer to a near-spherical, disc (``DiscPotential``), or
  multi-component (``MultiComponentPotential``) system, which is dominated by
  a different mix of families entirely.
* **The ``classify()`` parameters.** The training labels are
  ``classify()``'s own output; if you call ``classify_probabilistic()`` (or
  ``fit_probabilistic_classifier``) with non-default ``circ_thresh``,
  ``diffusion_threshold`` or ``inner_outer_ratio``, pass the *same* values
  used to build any ``classification=`` you supply, and refit if you change
  them later -- an old model is calibrated against thresholds that no longer
  match.

How it works
-------------

``classify_orbits_probabilistic`` / ``fit_probabilistic_classifier``:

#. Run (or reuse) ``classify_orbits`` to get deterministic labels.
#. Build a small reduced feature vector per orbit -- circulation, planarity,
   resonance order, frequency diffusion rate, the two frequency ratios and
   the x-tube inner/outer morphology ratio, all already computed by
   ``classify_orbits`` itself (log-transformed where the raw quantity spans
   decades, so classes that trace lines or points in frequency-ratio space
   are closer to elliptical in the fitted space).
#. Keep only the orbits confidently away from every *continuous* threshold
   ``classify_orbits`` applies (circulation, diffusion, the x-tube ratio) --
   this is the self-supervised training set.
#. Fit one Gaussian per class (full covariance where there is enough data,
   diagonal for smaller classes, skipped entirely below a handful of
   exemplars) with **uniform priors** across classes, so a physically rare
   family (an outer long-axis tube, say) is not penalised just for being
   numerically rare in the training population.
#. Evaluate every orbit's posterior via Bayes' rule.

Posterior probabilities are recomputed on demand from a cached per-orbit
feature row and the (tiny) fitted model, rather than stored as a population-
wide ``(n_orbits, n_classes)`` matrix -- ``probability_bar``/
``probabilities_for`` cost microseconds per query regardless of population
size, and the routinely-stored per-orbit summary is just the MAP label, its
probability and the posterior's entropy.

Caveats
-------

* **This cannot discover a class boundary the deterministic classifier gets
  wrong.** It is trained on ``classify()``'s own labels, so it only
  quantifies confidence *around* the boundaries that classifier already
  draws, and will agree with it almost everywhere away from those
  boundaries. Treat ``map_probability``/``entropy`` as a confidence
  diagnostic, not an independent check on the physics.
* **Spectral-only IRREGULAR orbits have no matching feature.** An orbit can
  be labelled irregular purely by the spectral (needs more than three base
  frequencies) test, which has no continuous analogue in the reduced feature
  set. Such an orbit's diffusion rate can look perfectly regular, so it will
  typically get a *confident* posterior for whichever regular family it
  otherwise resembles, not a low-confidence one. Do not read a high
  ``map_probability`` as ruling out this kind of irregularity -- cross-check
  against ``classify()``'s own ``labels`` if that distinction matters to you.
* **Unequal class variances can shift a boundary.** Two Gaussians with
  different spreads (e.g. a tightly clustered family next to a more diffuse
  one) cross at a point that is not simply the midpoint of their means, so
  the model's boundary between two classes need not sit exactly where
  ``classify()``'s own hard cut does. This is expected behaviour for a
  per-class Gaussian fit, not a bug.
* **Before trusting the numbers, calibrate.** Check that particles the model
  assigns a probability of roughly :math:`p` are actually correct roughly
  :math:`p` of the time, e.g. against a held-out population or the
  reproducible reference orbits used in ``tests/test_classify.py``.

.. seealso::

   :doc:`probabilistic` for the full API reference.
