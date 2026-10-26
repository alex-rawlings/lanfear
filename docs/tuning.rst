Choosing parameters and checking convergence
=============================================

Three settings decide how trustworthy a classification is: the potential orders
``n_max``/``l_max``, the integration length, and the chaos threshold
``diffusion_threshold``. Each has a helper.

Calibrated defaults and validated range
---------------------------------------

The classification thresholds were calibrated against equilibrium perfect
ellipsoids whose orbit families are known exactly (see `Reference data for
classifier development`_): four static shapes from prolate to near-oblate,
and a triaxial one tumbling about its short axis with corotation at 16, 8 and 4
half-mass radii. The defaults:

.. list-table::
   :header-rows: 1
   :widths: 24 14 62

   * - Parameter
     - Default
     - Basis
   * - ``n_periods``
     - 50
     - Accuracy is best at 25--50 periods and falls with longer windows
       (`Choosing the integration length`_).
   * - ``circ_thresh``
     - 0.9
     - Box vs tube. 0.5 is slightly better static (balanced accuracy
       0.97--0.995 against 0.966--0.986), 0.9 clearly better tumbling (0.96 /
       0.94 / 0.88 against 0.94 / 0.92 / 0.74 at corotation radii 16 / 8 / 4).
   * - ``hole_thresh``
     - 0.03
     - Applied about the pattern-speed axis of a rotating figure only, so that
       precessing boxes stay boxes; static labels are unaffected.
   * - ``inner_outer_ratio``
     - adaptive (fallback 0.84)
     - Otsu cut on the population's long-axis tubes; balanced accuracy
       0.92--0.98 on the static shapes, against 0.81--0.98 for a fixed 0.84.
   * - ``resonance_tol``, ``resonance_max_order``
     - 1e-3, 5
     - Boxlet purity ~80% (`Choosing resonance_tol`_).
   * - ``shared_line_tol``
     - 0.1
     - Rotating figures only: removes the 1:1 "boxlets" that are a box's
       precession.
   * - ``diffusion_threshold``, ``diffusion_drop``
     - 0.1, 0.5
     - Flags 24--41% of AGAMA-chaotic orbits at 1.3--4.8% false positives
       (`How much chaos the flags catch`_).
   * - ``irregular_amp_frac``
     - 0.1
     - Spectral test: 1.1% false positives on the static model.
   * - ``corotation_band``
     - ``(0.5, inf)``
     - Labels with ``epsilon`` below 0.4 are 94--98% right, at and beyond
       corotation 43--60%.

``freq_tol``, ``pattern_lock_tol``, ``winding_thresh``, ``libration_hole``
and ``pattern_tol`` were set on pinned and synthetic orbits, not against an
exact reference.

Family accuracy with these defaults (all orbits / those AGAMA finds regular):

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Model
     - All orbits
     - Regular orbits
   * - Static (triaxial, prolate, oblate, round)
     - 95.8, 91.4, 93.8, 90.6%
     - same (no chaos)
   * - Tumbling, corotation at 16 half-mass radii
     - 93.9%
     - 96.1%
   * - Tumbling, corotation at 8 half-mass radii
     - 92.3%
     - 95.5%
   * - Tumbling, corotation at 4 half-mass radii
     - 83.7%
     - 87.7%

The defaults are therefore **validated for static figures and for slow
tumbling, with corotation at 8 half-mass radii or beyond**, the regime of
tumbling ellipticals. With corotation at 4 half-mass radii they still work but
are clearly less reliable. Fast, bar-like rotators (corotation at 1--2
half-mass radii) are covered only by pinned orbits in the tests
(``tests/test_classify.py``, ``tests/test_rotating_classification.py``) and by
the run-to-run comparisons of ``make_rotation_grid.py``, not by an exact
reference. Validate them on your own model before relying on bar-family
fractions.

Choosing n_max / l_max
----------------------

``scripts/sweep_truncation.py`` sweeps a grid of ``(n_max, l_max)``, building and
validating the potential at each point (one MPI rank per combination), then
recommends the cheapest orders at the convergence knee. Feed the result to
``run_orbits_mpi.py``:

.. code-block:: bash

   srun --mpi=pmix -n 16 python scripts/sweep_truncation.py --file snap.hdf5
   # -> Recommended: n_max=8 l_max=4   ->  run_orbits_mpi.py --n-max 8 --l-max 4

It prints the full error grid and can save a heatmap with ``--plot sweep.png``.
The recommendation is convergence-relative (cheapest orders within ``--slack`` of
the best grid error), *not* an absolute median-error cut: the median potential
error is monopole-dominated, so an absolute cut would wrongly accept
``l_max = 0``.

From Python, ``Potential.truncation_convergence`` gives the same information as
a ``TruncationSweep`` with a ``.plot()`` method.

For a ``MultiComponentPotential``, each species is fit independently (see
:doc:`potentials`), so tune orders per species: build that species' own subset
(``ps.select(ps.species_mask("DM"))``) and sweep/validate it exactly as above
before fixing the order used in ``scf_component``/``disc_component``.

Choosing the integration length
-------------------------------

Longer windows do not converge on the true families. The same 10 000 orbits
of the triaxial perfect ellipsoid (static, and tumbling with corotation at 16
and 8 half-mass radii; see "Reference data" below) were integrated for 25, 50,
100, 200 and 400 periods at a fixed sampling interval:

* **Agreement with the true family falls as the window grows**: static 0.963
  (25 and 50 periods), 0.954, 0.941, 0.923 (400 periods); 0.966 to 0.937 and
  0.940 to 0.928 in the two tumbling models. The orbits lost are tubes that
  become boxes (at 50 to 400 periods, 318 short-axis and 107 long-axis tubes
  against 45 orbits going the other way): their angular momentum about the
  tube axis changes sign within 400 periods in the SCF fit, though not in the
  exact potential. The integration conserves the energy to ~1e-7, so this is
  the fitted potential's own dynamics: over long times its small departures
  from the true potential move orbits near a family boundary across it, and
  the fraction labelled diffusing grows too (0.6 to 1.5% static).
* **Labels keep changing up to 400 periods**: the family agrees with the
  400-period one for 0.85 / 0.87 / 0.89 / 0.91 of the orbits at 25 / 50 /
  100 / 200 periods (static; 0.77--0.84 tumbling), the shape for 0.93--0.98.
  Besides the tubes, boxes gain resonances as the frequency resolution
  improves (boxlets 185 at 50 periods, 775 at 400, static).
* In tumbling figures the agreement with the 400-period labels depends on
  ``epsilon``: 0.88--0.89 below 0.1, ~0.8 at 0.1--0.3, 0.57--0.78 at 0.3--1
  and ~0.6 beyond corotation. ``epsilon`` itself converges: from 50 to 400
  periods its median relative change is 0.5--1%, and 93--94% of orbits stay
  in the same ``epsilon`` bin. But no ``epsilon`` cut isolates the orbits
  whose labels change, which are mostly boxes (pibox, boxlet, irregular):
  ``near_corotation`` (``epsilon >= 0.5``) flags 6.5 / 13.5% of the orbits
  (corotation at 16 / 8 half-mass radii) and catches 14 / 24% of the changes;
  a cut at 0.3 would flag 9 / 21% and catch 17 / 38%. The band is therefore
  left at its accuracy-calibrated ``(0.5, inf)``.
* ``cls.unresolved`` marks orbits that are twice as likely to change (about
  half of them keep their family) but only 1--5% of the orbits, while 13--23%
  change; it does not isolate most of the changes.

So 25--50 periods give the best agreement with the true families on these
models, and a longer window does not reduce the error of family fractions: it
trades measurement noise for the fitted potential's long-term drift. The
uncertainty that remains is set by how well the potential is fitted, not by
the window length.

Extending drifting orbits (adaptive extension)
----------------------------------------------

A short integration cannot always tell chaos from a slowly converging regular
orbit: both show some frequency drift between the two halves of the run. The
distinction is how the drift behaves as the window grows. A regular orbit's
measured drift is a resolution effect and falls steeply (roughly as the length
ratio to the power -2), whereas a chaotic orbit's does not.

With ``max_extensions > 0``, ``analyse_family`` re-integrates only the orbits
whose diffusion rate exceeds ``diffusion_threshold``, for ``extension_factor``
times longer. An orbit whose rate has not fallen below ``diffusion_drop`` times
its previous value is judged chaotic and is not extended again; one that is still
above the threshold but falling is extended further, up to ``max_extensions``
times. Orbits below the threshold are never re-integrated, so the extra cost is
confined to the drifting orbits:

.. code-block:: python

   res = lf.analyse_family(pot, ps, n_periods=50,
                           max_extensions=2, extension_factor=2.0,
                           diffusion_threshold=0.1, diffusion_drop=0.5)
   res.n_periods_used       # (N,) periods each orbit was integrated for (50/100/200)
   res.diffusion_previous   # (N,) rate before the last extension (NaN if never extended)

   # Use the same values when classifying: an extended orbit is labelled irregular
   # only if its rate is above the threshold AND did not fall by the drop factor.
   cls = res.classify(diffusion_threshold=0.1, diffusion_drop=0.5)

Orbits never extended (all orbits when ``max_extensions=0``) are labelled by the
plain threshold cut. An orbit that is still above the threshold but falling when
the extensions run out keeps its regular label, and a warning reports how many
there are; raise ``max_extensions`` to settle them. Off by default. The run
script exposes it as ``--max-extensions``, ``--extension-factor``,
``--diffusion-threshold`` and ``--diffusion-drop``.

When to use it
^^^^^^^^^^^^^^

The evidence comes from the perfect ellipsoids of `Reference data for
classifier development`_: 10 000 orbits, static and tumbling with corotation
at 16 and 8 half-mass radii, integrated for 50 periods with and without
extension (2 rounds of factor 2, triggered at a diffusion rate of 0.02). The
families and the chaotic orbits (AGAMA Lyapunov exponents) are known exactly.

For extension:

* It tests the property that tells the two apart: a regular orbit's drift falls
  (roughly as the length ratio to the power -2) as the window grows, a chaotic
  orbit's does not.
* Fewer regular orbits are flagged at the default threshold: 1.76 to 1.50%
  static, 0.96 to 0.72% and 1.17 to 0.85% tumbling.
* At a fixed false-positive rate it catches 2--6 points more of the chaotic
  orbits, e.g. 48% at 1.0% false positives (threshold 0.05) against 42% at
  1.1% with the defaults, corotation at 8 half-mass radii.
* Only the drifting orbits are re-integrated: 5% static, 8% and 20% tumbling.

Against it:

* At the default threshold it catches no more chaos (0.18 to 0.16 and 0.42 to
  0.42 of the chaotic orbits): the gain only comes with a lower threshold.
* It cannot find chaos that sets in after the window: 31--43% of the chaotic
  orbits do not drift within 50 periods, so they are never extended. The
  irregular fraction stays a lower bound on the chaotic one.
* An extended orbit's label comes from the longer window, and longer windows
  move away from the true families (see `Choosing the integration length`_).
  The family accuracy of the extended orbits falls from 0.695 to 0.643 static,
  0.731 to 0.705 and 0.886 to 0.882 tumbling; over all orbits, by 0.1--0.25
  points.
* In the static perfect ellipsoid, which has essentially no chaos, 221 of the
  491 extended orbits are judged chaotic: the slow drift of the fitted
  potential itself.
* It costs 1.4, 2.3 and 3.3 times the integration time (static, corotation at
  16 and 8 half-mass radii). Near corotation it multiplies with the body-frame
  lengthening, so an orbit can be integrated for up to 32 times the window.
* Different orbits are measured over different windows, which complicates
  comparisons between snapshots and the half-window uncertainties of the
  probabilistic classification.

The balance depends on the system:

* **Static figures without a cusp or a black hole** (the perfect ellipsoid):
  there is almost no chaos to find (0.1% of the orbits), so extension only
  trims false positives. It is cheap, but gains nothing.
* **Tumbling figures**: chaos is common (12, 24 and 42% of the orbits with
  corotation at 16, 8 and 4 half-mass radii), so the recall gain is real, but
  so is the cost, and the extended orbits lie near corotation, where the labels
  are least reliable anyway.
* **Central black holes** (massive ellipticals, the main science case) make
  most centrophilic boxes chaotic. Static perfect ellipsoids with a central
  black hole of 0.2% and 1% of the stellar mass (``make_perfect_ellipsoid.py
  --bh-mass-fraction``; AGAMA Lyapunov exponents as the reference) have 28% and
  32% of their orbits chaotic: 78% and 91% of the boxes, 1% and 3% of the
  others. (A steep cusp alone does much the same: in a static squashed
  Hernquist model, cusp slope 1, 26% of the boxes are flagged, 40% with a 1%
  black hole.) The black hole does not hurt the families: they agree with the
  reference for 96.7% and 95.3% of the orbits (95.8% without a black hole).
  On 10 000 orbits integrated with and without extension:

  .. list-table::
     :header-rows: 1
     :widths: 46 27 27

     * -
       - BH 0.2%
       - BH 1%
     * - chaos setting in after 50 periods
       - 38% of the chaotic orbits
       - 21%
     * - orbits extended (boxes)
       - 13% (28%)
       - 24% (65%)
     * - integration time with extension
       - x1.4
       - x1.8
     * - family accuracy, without / with
       - 96.6 / 96.6%
       - 95.4 / 95.2%
     * - chaos recall at ``diffusion_threshold`` 0.1, without / with
       - 0.14 / 0.13
       - 0.33 / 0.33
     * - chaos recall at 0.03, without / with (regular orbits flagged)
       - 0.27 / 0.23 (3.6 / 2.6%)
       - 0.62 / 0.55 (2.9 / 2.1%)

  As in the tumbling figures, extension lowers both the recall and the false
  positives at a given threshold, and at a matched false-positive rate it
  gains only about 3 points of recall. Lowering the threshold gains far more:
  with the 1% black hole, 0.03 without extension catches 62% of the chaotic
  orbits (65% of the chaotic boxes) at 2.9% false positives, against 33% at the
  default 0.1. Chaos sets in sooner the more massive the black hole, so the
  lighter one leaves more of it out of reach of any threshold (0.27 at 0.03).
  Even then the flagged fraction (22% at 0.03, 11% at the default, for the 1%
  black hole) is a lower bound on the chaotic one (32%).

**Decision: leave extension off by default, including for black holes.** The
families and their fractions are unaffected by unflagged chaos (a chaotic orbit
keeps its shape) and are slightly worse with extension, at 1.4--3.3 times the
cost, and in every model tested (static, tumbling, with a black hole)
extension buys only 2--6 points of chaos recall at a matched false-positive
rate. When the chaotic fraction, or a selection of chaotic orbits, is itself the
result, as it often will be for the black-hole-fed (centrophilic, loss-cone)
orbits of a massive elliptical, **lower the classification threshold first**:

.. code-block:: python

   cls = res.classify(diffusion_threshold=0.03)   # no re-integration needed

``diffusion_threshold`` only enters the classification (unless extension is
on), so this needs no new integration. It roughly doubles the recall with a
black hole, at ~3% of the regular orbits flagged (3.4% in the static perfect
ellipsoid without one). Add extension only for the last few points:

.. code-block:: python

   res = lf.analyse_family(pot, ps, n_periods=50, max_extensions=2,
                           extension_factor=2.0, diffusion_threshold=0.02)
   cls = res.classify(diffusion_threshold=0.03, diffusion_drop=0.5)

Either way, treat the irregular fraction as a lower bound, and take the family
fractions from the classification at the default threshold.

Choosing resonance_tol
----------------------

A box orbit is a boxlet when its fundamental frequencies satisfy a low-order
commensurability ``|n . w| / max|w| < resonance_tol``. Non-resonant boxes also
pass this test by chance, more often the looser the tolerance and the higher
the order. ``scripts/boxlet_null_test.py`` measures this on your own results: it
shuffles each axis's frequencies independently between the box orbits (which
keeps the frequency distributions but destroys every real resonance) and
compares the hit rates of the real and shuffled frequencies per order and per
resonance vector:

.. code-block:: bash

   python scripts/boxlet_null_test.py orbits.npz --tol 1e-2 1e-3 5e-4

On the test models, ``1e-2`` (the former default) passed ~30% of the shuffled
triples at orders <= 5, so most of its "boxlets" were coincidences; at the
default ``1e-3`` ~3% pass, and 60--97% of the boxlets found at orders 2--5 are
genuine. Orders 6 and 7 are dominated by chance hits, so the generic search
stops at ``resonance_max_order = 5``, and the named pretzel (3:4, order 7) is
searched for explicitly; its purity depends on the population (96% in a
triaxial test model, ~50% in a bar-like one).

The default is close to the best balance. In a non-integrable static model (a
squashed Hernquist fit, axis ratios 1 : 0.8 : 0.6, with or without a central
black hole of 1% of the mass), boxes pile up at small residuals
``|n . w| / max|w|`` far above the chance level (below 1e-3, ~15 times the
chance estimate, against ~2--3 times in the integrable perfect ellipsoid).
These trapped orbits are spread widely in residual: their excess peaks near
1e-3 and extends past it, so they do not sit deep inside the tolerance (the
pinned test orbits, at 1e-4 to 1e-8, are cleaner than a population). The
boxlet fraction there is 9.6 / 17.6 / 24.1 / 32.4% of the regular boxes at
2e-4 / 5e-4 / 1e-3 / 2e-3, against a chance background of ~1--2 / 3 / 5 /
8--10% (perfect ellipsoids): the purity is ~80% from 2e-4 to 1e-3 and lower
above, while tightening to 5e-4 or 2e-4 loses about a quarter or a half of the
real boxlets.

The shuffle underestimates the chance rate of a real population: an orbit's
three frequencies are correlated (a smooth frequency map piles orbits up along
some ratios, e.g. the near-harmonic core), and a real box's frequencies are
ordered, which shuffling breaks. The perfect ellipsoids (below) are a cleaner
reference: their potential is integrable, so they have no resonant boxlet
families, and every boxlet lanfear finds there is a chance near-resonance. At
the default ``1e-3`` that background is 3.2--6.5% of the regular boxes
(triaxial 4.7%, prolate 6.5%, oblate 3.2%, round 3.9%), and it scales with the
tolerance (triaxial: 8.2 / 4.7 / 2.6 / 1.2% at 2e-3 / 1e-3 / 5e-4 / 2e-4). A
boxlet fraction of that order is no evidence of resonant families. 34--52% of
these boxlets are more centrophobic than 90% of the other boxes: over a
50-period window an orbit that close to a resonance traces the resonant
orbit's shape, so the label means "commensurate over the window".

In a rotating figure a box precessing about the pattern-speed axis has one
spectral line leading both in-plane coordinates (the frame rotation mixes
them), which reads as an exact 1:1 resonance. ``classify()`` therefore never
labels a boxlet an orbit whose two in-plane fundamentals agree to less than
``shared_line_tol`` (0.1) frequency resolutions; fewer than 0.5% of static
boxes agree that closely, against 7--15% in the tumbling models. This took the
tumbling perfect ellipsoids' boxlet fractions from 13.6 / 15.1 / 27.1% to 5.2 /
6.6 / 4.7% (corotation at 16 / 8 / 4 half-mass radii), the same as the static
background and scaling with the tolerance like it.

Choosing diffusion_threshold and diffusion_drop
-----------------------------------------------

``classify()`` labels orbits whose frequency-diffusion rate exceeds
``diffusion_threshold`` (default 0.1) as irregular; with extension the same
threshold decides which orbits are re-integrated, and ``diffusion_drop`` (default
0.5) decides whether an extended orbit's drift has fallen enough to be regular.
The right values depend on the integration length and population, so run the
helper on a saved result:

.. code-block:: bash

   python scripts/choose_diffusion_threshold.py orbits.npz --plot threshold.png

It prints the ``log10(rate)`` distribution at the base integration length, and a
table of how many orbits (and how much of each regular family, tubes especially)
each candidate threshold would flag; with extension this is the fraction that
would be re-integrated. It suggests a threshold two ways: the Otsu split of the
log-rate histogram, and the threshold that best agrees with the independent
spectral irregular test (F1). For results with extended orbits it also tabulates
candidate drop factors from the ratio of each orbit's rate after to before
extension, which is usually more clearly bimodal than the raw rate. It warns when
a distribution is not bimodal, in which case no value separates two populations
cleanly and a conservative one is safer.

Example: reading the figure
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--plot`` saves a figure like the one below, from a run with extension on a
large population (several hundred thousand orbits).

.. image:: figs/diffuse_plot.png
   :alt: Left, histogram of log10 diffusion rate at the base length with two suggestion lines. Right, histogram of the log10 rate ratio after and before extension with one suggestion line.

*Left: diffusion rate at the base integration length. Right: the rate after
extension divided by the rate before it, for the extended orbits. Dotted lines
mark the candidate values tabulated by the script; solid lines mark its
suggestions.*

**Left panel (choosing** ``diffusion_threshold`` **).** Almost all orbits sit in
one large peak at ``log10(rate)`` of about -3.7, a rate of order ``2e-4``: these
are the regular orbits, whose drift is only the resolution limit of a finite
window. A second, much smaller bump appears at about -0.5 (a rate of ~0.3),
well separated from the main peak by a shallow valley near -1.5 to -1 (rates of
0.03 to 0.1). The bump is the population that really drifts, so the threshold
belongs in the valley, not inside either peak. Here that is roughly 0.03 to 0.1.
The default of 0.1 sits on the lower flank of the bump and so leaves out its
weakest members; 0.03 catches more of the bump at the cost of a few regular
orbits from the main peak's tail (the per-family table shows how many).

Neither suggestion line is in the valley, and the script's note that they
differ by more than a decade applies:

* The **Otsu** line (green, 0.00087) is inside the main regular peak. With one
  dominant population and a small one, Otsu splits the large peak instead of
  finding the valley between the two, so it is not a usable answer here.
* The **spectral match** line (red, 0.57) lies to the right of the bump. The
  spectrally irregular population is too small to be visible at this scale
  (the orange bars cannot be seen), so the F1 score is driven by very few orbits
  and this value is not reliable either.

When the two suggestions disagree like this, trust the histogram and the
per-family table over the suggestions.

**Right panel (choosing** ``diffusion_drop`` **).** This shows how much each
extended orbit's drift changed when it was re-integrated for longer. It is
clearly bimodal, which is what makes extension useful:

* The large peak at about -2 (the rate fell by a factor of ~100) is orbits whose
  drift was a short-window effect. They settled as the window grew, which is
  what a regular orbit does, so they keep their regular label.
* The smaller peak at about 0 (a ratio of 1) is orbits whose drift did not
  change at all with the longer window. Drift that does not shrink with the
  window is the signature of chaos, so these are labelled irregular.

The valley between the two is at about -0.6 to -0.5 (a ratio of ~0.25 to 0.3),
so any ``diffusion_drop`` in the range 0.25 to 0.5 separates them well; the
default of 0.5 is safe. The red **Otsu** line (0.062) again falls on the
shoulder of the large peak, inside the regular population, so it would call some
falling orbits chaotic and should not be used. The drop factor is well
determined here: the two peaks are far apart and the valley between them is
shallow in comparison.

The caveats on this example: the figure alone does not show the integration
length or the extension factor, which set how far a regular orbit's rate falls
(roughly the length ratio to the power -2), so the position of the large peak on
the right depends on them. Read the peaks and valley off your own histograms, not
the numbers here.

How much chaos the flags catch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tumbling perfect ellipsoids (below) come with AGAMA Lyapunov exponents from
200-period integrations in the exact potential: 12%, 24% and 42% of their orbits
are chaotic at corotation radii of 16, 8 and 4 half-mass radii, and 0.1% in the
static model. With the defaults lanfear flags (irregular or diffusing) 24%, 35%
and 41% of those chaotic orbits, and 1.3%, 1.3% and 4.8% of the regular ones
(1.7% static, where every orbit is regular). **The irregular fraction is
therefore a lower bound on the chaotic fraction, by a factor of about 2--4.**

- For 31--43% of the chaotic orbits chaos only sets in after 50 periods, so no
  indicator measured in a 50-period window can see it; even among those chaotic
  within 50 periods only 27--45% are flagged. Strongly chaotic orbits are
  caught more often (38--56%) than weakly chaotic ones (9--31%).
- No single cut separates the two: the diffusion rate of chaotic orbits has a
  median of 0.016--0.035, that of regular ones 0.002 with a tail to ~0.02
  (regular outer long-axis tubes, which complete few long-axis cycles, have the
  widest tail). Lowering ``diffusion_threshold`` trades purity for recall
  roughly along (recall / regular orbits flagged, corotation at 8 half-mass
  radii) 0.35 / 1.3% at 0.1, 0.43 / 1.8% at 0.05, 0.51 / 2.5% at 0.03 and
  0.71 / 8.6% at 0.01. Measuring the drift in units of the frequency
  resolution does not separate them better.
- Adaptive extension (``max_extensions=2``) moves along much the same curve,
  gaining 2--6 points of recall at a fixed false-positive rate (e.g. 0.48 /
  1.0% at a threshold of 0.05) for about twice the integration time of a
  rotating figure.
- The spectral test alone catches 11--17% of the chaotic orbits at 0.6--1.7%
  false positives; loosening ``irregular_tol`` to 0.01 raises both (29--43%,
  2.5--6%).

Chaotic orbits that are not flagged keep their shape label, so the family
fractions are unaffected; every regular orbit that is flagged loses its family
label. The defaults favour purity; lower the threshold (and consider
extension) when the chaotic fraction itself is the quantity of interest.


Reference data for classifier development
-----------------------------------------

Two scripts produce fixed reference data for checking changes to the
classifier itself.

``scripts/make_classification_baseline.py`` integrates and classifies a static
triaxial population (an SCF fit to a squashed Hernquist sample; 20000 orbits by
default) and the pinned orbits of the test suite, and saves their labels and
diagnostics. Everything is regenerated from seeds stored in the baseline, so
``--compare`` re-runs the same population with the current code and reports the
label agreement, the transitions between families and whether every pinned
orbit kept its label (exit status 1 if not):

.. code-block:: bash

   python scripts/make_classification_baseline.py --out baseline_dir
   python scripts/make_classification_baseline.py --compare baseline_dir/baseline.npz

``scripts/make_rotation_grid.py`` integrates rotating figures with *known*
pattern speeds: a triaxial (``b/a = 0.79``, ``c/a = 0.5``) and a bar-like
(``b/a = 0.45``, ``c/a = 0.35``) ellipsoidal Hernquist model, tumbling about the
short axis with corotation at 16, 8, 4, 2, 1.5 and 1.2 half-mass radii, plus
the static case. The potential rotates at the true pattern speed; the
single-snapshot estimate and its uncertainty are stored in a ``.json`` file
next to each ``OrbitResults`` archive. Every run of a model integrates the same
particles, so labels can be compared orbit by orbit across pattern speeds. It
runs under MPI:

.. code-block:: bash

   srun -n 8 python scripts/make_rotation_grid.py --out-dir grid_dir

``scripts/make_perfect_ellipsoid.py`` (run in an environment with AGAMA)
builds an equilibrium N-body model of a triaxial perfect ellipsoid with
Schwarzschild's method and records the exact orbit family of every particle:
the perfect ellipsoid's potential is of Staeckel form, so box, short-axis tube
and inner and outer long-axis tubes are fixed by the integrals of motion, read
off from which ellipsoidal-coordinate boundaries each orbit reaches.
``scripts/compare_perfect_ellipsoid.py`` runs lanfear on the snapshot and prints
confusion matrices against those families, and scans ``circ_thresh`` and
``inner_outer_ratio``. On a model with axis ratios 0.79 and 0.5, lanfear's
families agree with the exact ones for 95.8% of orbits. Inner and outer
long-axis tubes are separated by the frequency ratio
``|w_x| / sqrt(|w_y| |w_z|)``: outer tubes sit at ~0.95 for any shape, inner
ones lower by a shape-dependent amount, so the best cut moves from ~0.79
(prolate) to ~0.90 (near-oblate). By default ``classify()`` therefore places
the cut from the population itself (Otsu's method on its long-axis tubes'
ratios, ``lf.adaptive_inner_outer_cut``), and falls back to 0.84 with a
warning when the population does not look like a mix of the two families. On
four perfect ellipsoids from prolate to near-oblate this gives 92--98%
balanced accuracy (a fixed 0.84: 81--98%; a shape-based test: ~70%). The cut
used is ``cls.inner_outer_ratio``; pass ``inner_outer_ratio=`` to fix it, e.g.
to classify several snapshots with the same cut.

In tumbling figures the adaptive cut is less reliable. Corotation at 16
half-mass radii still gives an accepted cut (0.885; balanced accuracy 0.988,
against 0.997 at the best cut, 0.850). At 8 and 4 half-mass radii it is rejected
and the fixed 0.84 is used (0.858 at 8, where the best, 0.845, gives 0.962;
too few inner tubes at 4 to score). Three things cause this: two thirds of the
orbits lanfear calls long-axis tubes are, in the reference, chaotic or
ambiguous orbits at ratios 0.89--1.04 that pull Otsu's split upwards; for
~15% of the long-axis tubes the leading x line is the circulation frequency
shifted by the pattern speed, putting their ratio near 1.03 (they are still,
correctly, outer tubes); and inner and outer tubes move closer (medians 0.83
and 0.91 at 8 half-mass radii), so with few inner tubes the best cut is not
measurable. Inner long-axis tubes are rare there (0.6% and 0.1% of the mass at
16 and 8 half-mass radii), so the effect on the family fractions is small;
relaxing the acceptance only for rotating figures gains a few orbits and was
not adopted.

The same comparison, on tumbling figures, sets the box/tube threshold
``circ_thresh`` (default 0.9) and the hole cut ``hole_thresh`` (default 0.03).
In a figure tumbling about its short axis a box precesses steadily, so its
angular momentum about that axis keeps one sign most of the time
(``circ_z`` ~0.7--0.8), but it still passes through the centre (hole ~0.02 of
its mean radius, against ~0.3 for a short-axis tube). These are the orbits that
reach a central black hole, so they must stay boxes. Box-vs-tube balanced
accuracy with the hole cut, corotation at 16 / 8 / 4 half-mass radii: 0.94 /
0.92 / 0.74 at ``circ_thresh`` 0.5 and 0.96 / 0.94 / 0.88 at 0.9. Without the
hole cut family accuracy at the defaults falls from 94 / 92 / 84% to 94 / 90 /
76%. On the static models 0.5 is slightly better (0.97--0.995 against
0.966--0.986), and there the hole cut only removes thin tubes, so it is applied
only about the rotation axis of a rotating figure and leaves static labels
unchanged. Family accuracy with the defaults (all orbits / regular ones): 95.8%
static (triaxial); 93.9 / 96.1%, 92.3 / 95.5% and 83.7 / 87.7% tumbling. The tubes called boxes in a
static figure are thin, near-radial ones: their central hole (~0.05 kpc) is
comparable to the centring error of ``prepare()``, and no threshold recovers
them.

The same runs set the corotation band of ``cls.near_corotation`` (default
``(0.5, inf)``). Pooled over the three tumbling models, the families of orbits
with ``epsilon = |Omega_p| / |Omega_phi|`` below 0.4 agree with the reference
for 94--98% of them, falling through 0.84--0.89 at 0.4--0.6 to ~0.6 at
corotation and 0.43--0.58 beyond it, with no recovery out to the largest
``epsilon`` (orbits the pattern outruns). With the band open-ended, the unflagged
orbits are 95% right (static model: 96%), against 92% for the earlier
``(0.5, 2.0)``; it flags 6%, 10% and 25% of the orbits at corotation radii of
16, 8 and 4 half-mass radii. Orbits that do not wind about the pattern axis
have no ``epsilon`` and are never flagged; they are 97% right at 16 and 8
half-mass radii but 75% at 4.

Outer long-axis tubes are the least well recovered family (recall 0.70--0.88
on the static models). Of those missed in the triaxial model, half are called
boxes and a sixth short-axis tubes. These are thin orbits next to a family
boundary (their hole about x is as small as a box's): re-integrated in the same
SCF fit they get the same labels, but in a higher-order fit (``n_max`` 32,
``l_max`` 12) about a third of them, and about 12% of the correctly labelled
outer tubes, change family, so the higher order does not recover more of them
overall. Their family is set by the accuracy of the fitted potential, not by
the classifier, and no threshold on the classifier diagnostics fixes it. The
rest are called rosettes: they orbit beyond ~100 scale radii, where the
figure is nearly spherical and their orbit plane does not precess measurably
in 50 periods (frequency differences of ~0.4%, below the window's resolution
of ~1.6%).

Rosettes have no Staeckel family of their own, so the comparison also scores
each one as the tube about its circulation axis (``cls.tube_axis``; about x,
inner or outer by the classification's own frequency-ratio cut). This raises
the accuracy from 95.8 / 91.4 / 93.8 / 90.6% to 96.5 / 92.8 / 94.8 / 92.9%
(triaxial / prolate / oblate / round), and in the tumbling models from 93.9 /
92.3 / 83.7% to 95.5 / 93.7 / 85.7% (corotation at 16 / 8 / 4 half-mass
radii). In the tumbling models nearly every rosette folds onto its reference
family. In the static ones the rosettes that are outer long-axis tubes fold
correctly too (144 of 148, triaxial), but only about a third of those that
are short-axis tubes do: the rest are far-out loops tilted well away from the
short axis, whose ``L_z`` keeps its sign while they circulate more strongly
about x or y. They fold to outer long-axis or intermediate-axis tubes, which
lowers the outer long-axis tubes' precision (0.85 to 0.75, triaxial). A
rosette's circulation axis is therefore a reliable family only in a tumbling
figure, where the frame makes its orbit plane precess about z. To run the
comparison:

.. code-block:: bash

   python scripts/make_perfect_ellipsoid.py --out-dir pe_dir      # AGAMA env
   python scripts/compare_perfect_ellipsoid.py --dir pe_dir       # lanfear env

``make_perfect_ellipsoid.py --corotation-radius R`` builds a figure tumbling
about its short axis with corotation at ``R`` half-mass radii. The reference
families then come from the body-frame circulation of long integrations in the
exact potential, with AGAMA Lyapunov exponents marking chaotic orbits;
``--bh-mass-fraction f`` adds a central black hole of ``f`` times the stellar
mass (a softened point mass in the potential and a PartType5 particle in the
snapshot), with the reference built the same way. On a
cluster, ``scripts/submit_perfect_ellipsoid.sh`` runs both steps as one slurm
job (from ``scripts/``):

.. code-block:: bash

   sbatch submit_perfect_ellipsoid.sh pe_dir                # static
   sbatch submit_perfect_ellipsoid.sh pe_rcr16 16           # tumbling

``scripts/compare_circulation.py`` reads such a grid and follows each orbit's
static family through the rotating runs, printing per family the medians of the
angular-momentum circulation, the winding circulation, the
azimuthal-to-radial frequency ratio and the hole size, and the fraction a few
candidate tube criteria would call tubes, plus the pattern-locking
distribution:

.. code-block:: bash

   python scripts/compare_circulation.py --grid-dir grid_dir --figdir figures
