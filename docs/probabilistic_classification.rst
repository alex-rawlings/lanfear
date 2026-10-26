Probabilistic classification
=============================

:doc:`usage`'s ``res.classify()`` labels each orbit by thresholding a handful
of diagnostics -- circulation, hole size, frequency ratios, resonance
residuals, ... -- measured over a finite integration window. Each of them has a
measurement error, so an orbit whose diagnostic lies within that error of a
threshold has a label that a slightly different measurement (a longer window,
other sampling) could change.

``res.classify_probabilistic()`` turns those errors into probabilities. For
every orbit and every label scheme of ``classify()`` (classes, families,
shapes, boxlet types, sense, pattern resonances, regularity) it gives **the
probability that the deterministic rule would give each label if the
diagnostics were known exactly**, given that orbit's own measurement noise. It
does not change ``classify()``: the most probable label is the deterministic
one except where more than half of the draws cross a threshold, and the new
information is how secure each label is. No model is fitted, so an orbit's
probabilities do not depend on the rest of the population.

Quickstart
----------

The orbit results need the half-window summary columns (written by
``analyse_family``/``analyse_states`` since they were added; older results must
be re-run):

.. code-block:: python

   import lanfear as lf

   res = lf.analyse_family(pot, ps, family="STAR", n_periods=50, n_lines=4)
   prob = res.classify_probabilistic()        # n_draws=32, seed=0; classify() kwargs pass through

   names, p = prob.probabilities("family")    # (N, K) probability of every family
   prob.map_labels("family")                  # (N,) most probable family (index)
   prob.map_names("family")                   # (N,) its name
   prob.map_probability("family")             # (N,) its probability
   prob.entropy("family")                     # (N,) in bits; 0 = every draw agrees
   prob.counts("family")                      # expected counts (sum of probabilities)
   prob.counts_uncertainty("family")          # their measurement-error spread
   prob.counts("family", particles=ps)        # mass-weighted

   names, p = prob.probability_bar(id=123456, scheme="shape")    # one orbit
   ax = prob.plot_probability_bar(id=123456, scheme="family")
   names, mean_p = prob.probabilities_for(label="pibox", scheme="family")

   ambiguous = prob.mask_ambiguous(0.8, scheme="family")     # P(most probable) < 0.8
   hard = prob.to_hard()        # an OrbitClassification with the most probable labels
   prob.base                    # the deterministic classification itself

``scheme`` is any of ``prob.schemes``: ``"class"`` (the default, the
``lf.OrbitClass`` labels), ``"family"``, ``"shape"``, ``"boxlet"``,
``"sense"``, ``"pattern_resonance"`` and ``"regularity"``. All schemes come from
the same draws, so they are consistent with each other.

How it works
------------

#. **Errors from the two halves of the window.**
   ``lf.measurement_uncertainties(res)`` compares the values each diagnostic
   takes in the two halves of the integration window: the angular-momentum
   circulation about each axis, the closest approach to each axis, and each
   axis's leading frequency (summary columns ``L*_mean_h1``,
   ``L*_abs_mean_h1``, ``rho_*_min_h1``/``_h2`` and ``freq_*_h1``/``_h2``).
   Half the difference is the error of the full-window value. The planar
   (azimuthal) and radial frequencies have no half-window values and take the
   orbit's largest axis-frequency error.
#. **Correlated draws.** The errors of one orbit are correlated: a resonant
   orbit stays resonant in both halves, so its frequencies differ between them
   along the resonance, not across it. Each draw therefore moves all of an
   orbit's half-window diagnostics together, by one Gaussian multiple of their
   half-differences (independent axis-by-axis errors would wrongly break
   every resonance). Only the planar and radial frequencies get independent
   deviates.
#. **The deterministic rules on every draw.** Each perturbed copy is
   classified by ``classify_orbits`` with the same settings, including the
   inner/outer cut placed on the original population, and the labels of every
   scheme are counted. A label's probability is its share of the
   ``n_draws`` draws (so probabilities are multiples of ``1 / n_draws``).

Each draw costs one classification of the population, so ``n_draws = 32``
takes about 32 times as long as ``classify()`` (still small next to the orbit
integration). The counts are stored as small integers, one per orbit and label.

What the probabilities mean, and what they leave out
----------------------------------------------------

A probability below 1 says the label rests on a diagnostic within measurement
error of a threshold. It is testable: integrating the same orbits for longer
should change the label of an orbit with probability ``p`` with a chance of
about ``1 - p``. On 10 000 orbits of the triaxial perfect ellipsoid (static,
and tumbling with corotation at 16 half-mass radii), re-integrated for 100
instead of 50 periods, the probabilities **rank** the orbits correctly but are
**overconfident**: the fraction of family labels the longer run keeps rises
from 0.45--0.57 (probability below 0.6) through 0.55--0.68 (0.6--0.95) and
0.78--0.81 (0.95--1) to 0.94--0.95 for the 88--90% of orbits with probability 1.
About a third of the changes among those certain orbits are regularity flips
(irregular to regular or back), which are not modelled; most of the rest are
boxes in which the finer frequency resolution of the longer window finds a
resonance (pibox to boxlet), a change with window length rather than noise
the two halves can reveal, and thin tubes near the box boundary.

They do not include:

* **Regularity.** The spectral lines and the diffusion rate are not
  perturbed, so regularity is that of ``classify()`` (except for the rare orbit
  whose spectral test responds to the perturbed frequency scale it uses as a
  tolerance). How much chaos the regularity flags catch is discussed in
  :doc:`tuning`.
* **Errors the two halves of one window cannot reveal**: how well the
  potential is fitted (thin tubes next to a family boundary can change family
  with the expansion order), chaos that sets in after the window, or a
  mismatch between a rule's definition and an orbit's true family.

So a probability of 1 means "secure against measurement noise", not "certainly
the true family"; the comparisons with reference families in :doc:`tuning`
measure the latter.

With a central SMBH binary, pass ``drop_binary_interacting=True`` to leave out
the orbits flagged as reaching the binary (see "SMBH binaries" in
:doc:`usage`); a ``classification`` passed in must then have been computed
without them too.

.. seealso::

   :doc:`probabilistic` for the full API reference.
