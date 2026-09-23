"""Bayesian posterior-probability layer on top of :mod:`lanfear.classify`.

:func:`lanfear.classify.classify_orbits` assigns each orbit a single family
from a handful of hard-thresholded, physically motivated tests (circulation,
resonance, frequency diffusion, ...). Many of those thresholds sit at a
genuinely ambiguous physical boundary -- an orbit near the circulation cut, or
near the chaos-diffusion cut, does not have a well-defined hard label even in
principle. This module adds an optional, additive posterior layer: a
per-class Gaussian fitted in the same *reduced* feature space
(:func:`feature_matrix`) that :func:`~lanfear.classify.classify_orbits`
already uses, giving each orbit a full posterior probability over the classes
instead of a single MAP label.

The classifier is **self-supervised**: it is trained on
:func:`~lanfear.classify.classify_orbits`'s own labels, restricted to orbits
that sit confidently away from every threshold that classifier uses
(:func:`fit_probabilistic_classifier`). It therefore cannot discover a class
boundary the deterministic classifier gets wrong -- it only quantifies
confidence *around* that boundary, which is the point: it never needs a
separate, independently labelled training set, and it stays consistent with
the validated deterministic logic by construction.

Priors are uniform across classes (a rare family, e.g. an outer long-axis
tube, is not penalised just for being outnumbered by boxes), so posterior
probabilities reduce to normalised class-conditional likelihoods.

Everything here operates on the same cheap, already-reduced per-orbit
quantities :func:`~lanfear.classify.classify_orbits` uses, so it costs nothing
next to orbit integration: fitting is a single closed-form pass over a
handful of features, and a single particle's posterior can be recomputed
on demand in microseconds from its cached feature row plus the (kilobyte-sized)
fitted model, so per-particle queries never require materialising an
``(n_orbits, n_classes)`` array.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

from ._logging import get_logger
from .classify import (
    CLASS_NAMES,
    OrbitClass,
    OrbitClassification,
    _colour_for,
    _latex_label,
)

logger = get_logger(__name__)

# The reduced feature vector fed to the Bayesian classifier -- the same
# quantities classify_orbits() uses, reparametrised so that classes which
# trace lines/points in the raw frequency-ratio space (see
# OrbitClassification.plot_frequency_map) are closer to elliptical blobs:
# ratios and rates are strictly positive and span decades, so they are used
# in log space; circulation and planarity are already O(1) and bounded.
FEATURE_NAMES: Tuple[str, ...] = (
    "circ_x",
    "circ_y",
    "circ_z",
    "planarity",
    "log_resonance_order",
    "log_diffusion_rate",
    "log_freq_ratio_x",
    "log_freq_ratio_y",
    "log_x_tube_ratio",
)

# Floors applied before taking a log, so a zero/negative raw value (no
# resonance found, a non-x-tube's meaningless ratio, ...) does not produce
# -inf. Chosen well below anything physically meaningful for each quantity.
_DIFFUSION_FLOOR = 1e-12
_XTUBE_RATIO_FLOOR = 1e-3
_XTUBE_RATIO_CEIL = 1e3  # x_tube_ratio reads 1e30 when there is no border data


def _check_aligned(classification: OrbitClassification, results) -> None:
    """Raise if ``classification`` and ``results`` do not describe the same orbits.

    Parameters
    ----------
    classification : lanfear.OrbitClassification
        A classification produced from ``results``.
    results : lanfear.OrbitResults
        The orbit results it should have been built from.

    Raises
    ------
    ValueError
        If the particle IDs do not match, in order.
    """
    if classification.ids is None or not np.array_equal(
        classification.ids, results.ids
    ):
        raise ValueError(
            "classification and results must describe the same orbits in the "
            "same order (mismatched or missing particle ids); build "
            "classification with classify_orbits(results, ...)."
        )


def feature_matrix(classification: OrbitClassification, results) -> np.ndarray:
    """Build the reduced feature matrix the Bayesian classifier trains/predicts on.

    Reuses exactly the quantities :func:`~lanfear.classify.classify_orbits`
    itself computes (circulation, planarity, resonance order, frequency
    diffusion, frequency ratios, the x-tube inner/outer morphology ratio), so
    no new orbit integration or trajectory data is needed -- this is a
    function of the same compact summary that makes classification itself
    cheap at millions of orbits.

    Parameters
    ----------
    classification : lanfear.OrbitClassification
        Deterministic classification, as returned by
        :func:`~lanfear.classify.classify_orbits`.
    results : lanfear.OrbitResults
        The orbit results ``classification`` was built from (for the raw
        diffusion rate and ``x_tube_ratio`` summary column, which are not
        carried on :class:`~lanfear.OrbitClassification`).

    Returns
    -------
    features : numpy.ndarray
        ``(n_orbits, len(FEATURE_NAMES))`` float64 array. Rows with a
        non-finite entry (e.g. an axis with no oscillation, so its frequency
        ratio is undefined) are left non-finite rather than imputed -- see
        :meth:`BayesianOrbitClassifier.predict_prob`.

    Raises
    ------
    ValueError
        If ``classification`` and ``results`` describe different orbits.
    """
    _check_aligned(classification, results)
    circ = classification.circulation  # (N,3)
    planarity = classification.planarity
    log_resonance = np.log10(classification.resonance_order.astype(float) + 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = results.frequency_ratios  # (N,2), NaN where undefined
        log_ratio_x = np.log(ratios[:, 0])
        log_ratio_y = np.log(ratios[:, 1])

        rate = results.diffusion_rate()
        log_diffusion = np.log10(np.clip(rate, _DIFFUSION_FLOOR, None))

        x_tube_ratio = results.column("x_tube_ratio")
        log_x_tube = np.log10(
            np.clip(x_tube_ratio, _XTUBE_RATIO_FLOOR, _XTUBE_RATIO_CEIL)
        )

    return np.column_stack(
        [
            circ[:, 0],
            circ[:, 1],
            circ[:, 2],
            planarity,
            log_resonance,
            log_diffusion,
            log_ratio_x,
            log_ratio_y,
            log_x_tube,
        ]
    ).astype(np.float64)


def _confident_training_mask(
    classification: OrbitClassification,
    results,
    circ_thresh: float,
    diffusion_threshold: Optional[float],
    inner_outer_ratio: float,
    min_circ_margin: float,
    min_diffusion_margin_dex: float,
    min_xtube_margin_dex: float,
) -> np.ndarray:
    """Orbits confidently away from every continuous threshold classify_orbits uses.

    Covers the three continuous decision boundaries classify_orbits makes:
    circulation (box vs tube, ``circ_thresh``), frequency diffusion (regular
    vs chaotic, ``diffusion_threshold``) and the x-tube inner/outer morphology
    ratio (``inner_outer_ratio``). IRREGULAR orbits are additionally required
    to have a diffusion rate confidently *above* ``diffusion_threshold``
    (not just far from it in either direction): an orbit can also be labelled
    IRREGULAR by the purely spectral (>3 base frequencies) test, which has no
    continuous analogue in the reduced feature set, so such an orbit's
    diffusion rate can look perfectly regular -- training on it would teach
    the model to associate the IRREGULAR class with features it does not
    actually have. The remaining discrete/combinatorial parts of
    classify_orbits (the resonance-lattice search, the y-tube frequency-lock
    corroboration) have no continuous distance either and are not
    margin-filtered; their instability shows up as extra scatter within a
    class's fitted covariance rather than as mislabelled training points near
    a sharp boundary. One consequence worth knowing: a spectral-only
    IRREGULAR orbit, having no distinguishing feature, will typically get a
    *confident* posterior for whichever regular family it resembles, not a
    low-confidence one -- this is a real representational limit of the
    feature set, not a calibration failure, and it means ``map_probability``
    should not be read as "probability of being genuinely regular" for
    orbits near a spectral (as opposed to diffusion) chaos boundary.

    Returns
    -------
    confident : numpy.ndarray
        (N,) boolean, True for orbits classify_orbits gave a real family to
        (not UNCLASSIFIED) and that sit at least the given margin away from
        every applicable threshold.
    """
    circ_margin = np.abs(np.max(classification.circulation, axis=1) - circ_thresh)
    confident = circ_margin >= min_circ_margin

    if diffusion_threshold is not None:
        rate = results.diffusion_rate()
        with np.errstate(divide="ignore", invalid="ignore"):
            diffusion_margin = np.abs(
                np.log10(np.clip(rate, _DIFFUSION_FLOOR, None))
                - np.log10(diffusion_threshold)
            )
        confident &= np.isfinite(diffusion_margin) & (
            diffusion_margin >= min_diffusion_margin_dex
        )
        # IRREGULAR can also be assigned by the spectral (>3 base frequencies)
        # test, unrelated to diffusion -- such an orbit can have a diffusion
        # rate confidently *below* diffusion_threshold, i.e. a training
        # exemplar whose only continuous feature for this decision
        # (log_diffusion_rate) looks indistinguishable from a regular orbit.
        # Restrict IRREGULAR training exemplars to ones the diffusion
        # criterion itself agrees with, so the fitted class is not diluted by
        # spectral-only outliers that share no feature signature with it.
        is_irregular = classification.labels == OrbitClass.IRREGULAR
        confident &= ~is_irregular | (rate > diffusion_threshold)

    is_x_tube = classification.tube_axis == 0
    x_tube_ratio = results.column("x_tube_ratio")
    with np.errstate(divide="ignore", invalid="ignore"):
        xtube_margin = np.abs(
            np.log10(np.clip(x_tube_ratio, _XTUBE_RATIO_FLOOR, _XTUBE_RATIO_CEIL))
            - np.log10(inner_outer_ratio)
        )
    confident &= ~is_x_tube | (xtube_margin >= min_xtube_margin_dex)

    confident &= classification.labels != OrbitClass.UNCLASSIFIED
    return confident


@dataclass
class BayesianOrbitClassifier:
    """A per-class Gaussian posterior classifier over the reduced orbit features.

    Fitted by :meth:`fit` (usually via :func:`fit_probabilistic_classifier`,
    which handles the self-supervised training-set construction). Priors are
    uniform across classes, so :meth:`predict_prob` is a normalised
    class-conditional likelihood, not a full Bayesian posterior with a
    learned prior -- deliberately, so a physically rare class is not
    penalised for being numerically rare in the training population.

    Parameters
    ----------
    class_labels : numpy.ndarray
        (n_classes,) :class:`~lanfear.OrbitClass` integer values covered by
        the model, ascending.
    means : numpy.ndarray
        (n_classes, n_features) fitted mean of each class.
    cholesky : numpy.ndarray
        (n_classes, n_features, n_features) lower-triangular Cholesky factor
        of each class's (regularised) covariance.
    log_det_cov : numpy.ndarray
        (n_classes,) precomputed ``log|Sigma|`` for each class.
    diagonal : numpy.ndarray
        (n_classes,) bool, whether that class had too few confident training
        exemplars for a full covariance and fell back to a diagonal one.
    n_training : numpy.ndarray
        (n_classes,) number of confident training exemplars used per class.
    feature_names : tuple of str
        Names of the features in ``means``' second axis, for reference.
    """

    class_labels: np.ndarray
    means: np.ndarray
    cholesky: np.ndarray
    log_det_cov: np.ndarray
    diagonal: np.ndarray
    n_training: np.ndarray
    feature_names: Tuple[str, ...] = FEATURE_NAMES
    _format: str = field(default="lanfear.BayesianOrbitClassifier", repr=False)
    _version: int = field(default=1, repr=False)

    @property
    def class_names(self) -> Dict[int, str]:
        """Label-to-name mapping for the classes this model covers.

        Returns
        -------
        names : dict
            Mapping from :class:`~lanfear.OrbitClass` integer value to its
            lower-case name, for the classes in :attr:`class_labels`.
        """
        return {int(k): CLASS_NAMES[int(k)] for k in self.class_labels}

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        reg_covar: float = 1e-6,
        min_samples_full: Optional[int] = None,
        min_samples_diag: int = 5,
    ) -> "BayesianOrbitClassifier":
        """Fit one Gaussian per class from labelled feature rows.

        Parameters
        ----------
        features : numpy.ndarray
            (n_orbits, n_features) training features, e.g. from
            :func:`feature_matrix`. Rows with a non-finite entry are dropped.
        labels : numpy.ndarray
            (n_orbits,) :class:`~lanfear.OrbitClass` integer label per row.
        reg_covar : float, optional
            Ridge added to each class's covariance diagonal, for numerical
            stability (small classes especially can be near-singular).
        min_samples_full : int, optional
            Minimum confident exemplars needed to fit a full covariance for a
            class; below this it falls back to a diagonal one. Defaults to
            twice the number of features (a standard rule of thumb for a
            well-conditioned sample covariance).
        min_samples_diag : int, optional
            Minimum confident exemplars needed to fit even a diagonal
            covariance; classes with fewer are skipped entirely (with a
            warning) rather than fit from too little data.

        Returns
        -------
        classifier : BayesianOrbitClassifier
            The fitted model, covering every class with at least
            ``min_samples_diag`` confident exemplars.

        Raises
        ------
        ValueError
            If no class had enough confident exemplars to fit.
        """
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels)
        finite = np.all(np.isfinite(features), axis=1)
        if not np.all(finite):
            logger.warning(
                f"Dropping {int(np.sum(~finite))} of {len(finite)} training rows "
                "with a non-finite feature (e.g. an axis with no oscillation)."
            )
        features, labels = features[finite], labels[finite]
        n_features = features.shape[1]
        if min_samples_full is None:
            min_samples_full = 2 * n_features

        means, chols, log_dets, diag_flags, counts, kept = [], [], [], [], [], []
        for k in sorted(int(v) for v in np.unique(labels)):
            rows = features[labels == k]
            n = len(rows)
            name = CLASS_NAMES.get(k, str(k))
            if n < min_samples_diag:
                logger.warning(
                    f"Skipping class {name!r}: only {n} confident training exemplars "
                    f"(< {min_samples_diag}); no probabilistic model fitted for it."
                )
                continue
            mean = rows.mean(axis=0)
            centred = rows - mean
            is_diag = n < min_samples_full
            if is_diag:
                logger.info(
                    f"Class {name!r}: {n} confident exemplars (< {min_samples_full}), "
                    "using a diagonal covariance."
                )
                cov = np.diag(centred.var(axis=0, ddof=1) + reg_covar)
            else:
                cov = (centred.T @ centred) / (n - 1) + reg_covar * np.eye(n_features)
            chol = np.linalg.cholesky(cov)
            means.append(mean)
            chols.append(chol)
            log_dets.append(2.0 * np.sum(np.log(np.diag(chol))))
            diag_flags.append(is_diag)
            counts.append(n)
            kept.append(k)

        if not kept:
            raise ValueError(
                "No class had enough confident training exemplars to fit "
                f"(need >= {min_samples_diag}); loosen the margins or supply a "
                "larger population."
            )

        return cls(
            class_labels=np.array(kept, dtype=np.int64),
            means=np.stack(means),
            cholesky=np.stack(chols),
            log_det_cov=np.array(log_dets),
            diagonal=np.array(diag_flags),
            n_training=np.array(counts, dtype=np.int64),
        )

    def predict_log_likelihood(self, features: np.ndarray) -> np.ndarray:
        """Per-class Gaussian log-likelihood of each row.

        Parameters
        ----------
        features : numpy.ndarray
            (n_orbits, n_features) feature rows, e.g. from
            :func:`feature_matrix`.

        Returns
        -------
        log_likelihood : numpy.ndarray
            (n_orbits, n_classes) log N(x | mean_k, cov_k); NaN in a row where
            ``features`` had a non-finite entry.
        """
        x = np.asarray(features, dtype=np.float64)
        n_orbits = x.shape[0]
        n_classes = len(self.class_labels)
        n_features = self.means.shape[1]
        out = np.full((n_orbits, n_classes), np.nan)
        finite = np.all(np.isfinite(x), axis=1)
        if not np.any(finite):
            return out
        xf = x[finite]
        const = n_features * np.log(2.0 * np.pi)
        for j in range(n_classes):
            diff = (xf - self.means[j]).T  # (n_features, m)
            y = solve_triangular(self.cholesky[j], diff, lower=True)
            maha = np.sum(y**2, axis=0)
            out[finite, j] = -0.5 * (const + self.log_det_cov[j] + maha)
        return out

    def predict_prob(self, features: np.ndarray) -> np.ndarray:
        """Posterior class probability of each row (uniform priors).

        Parameters
        ----------
        features : numpy.ndarray
            (n_orbits, n_features) feature rows, e.g. from
            :func:`feature_matrix`.

        Returns
        -------
        proba : numpy.ndarray
            (n_orbits, n_classes) posterior probabilities, summing to 1 along
            axis 1; a row is all-NaN where ``features`` had a non-finite
            entry. Columns are ordered as :attr:`class_labels`.
        """
        log_likelihood = self.predict_log_likelihood(features)
        out = np.full_like(log_likelihood, np.nan)
        finite_rows = np.all(np.isfinite(log_likelihood), axis=1)
        if np.any(finite_rows):
            m = log_likelihood[finite_rows]
            log_norm = logsumexp(m, axis=1, keepdims=True)
            out[finite_rows] = np.exp(m - log_norm)
        return out

    def predict(
        self, features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MAP label, its posterior probability, and the posterior's entropy.

        Parameters
        ----------
        features : numpy.ndarray
            (n_orbits, n_features) feature rows, e.g. from
            :func:`feature_matrix`.

        Returns
        -------
        labels : numpy.ndarray
            (n_orbits,) MAP :class:`~lanfear.OrbitClass` value, or ``-1``
            where the features could not be evaluated (a non-finite entry).
        map_probability : numpy.ndarray
            (n_orbits,) posterior probability of the MAP label (NaN where
            unevaluable).
        entropy : numpy.ndarray
            (n_orbits,) Shannon entropy of the posterior, in bits (0 = fully
            confident, ``log2(n_classes)`` = uniform over every class; NaN
            where unevaluable).
        """
        proba = self.predict_prob(features)
        n_orbits = proba.shape[0]
        labels = np.full(n_orbits, -1, dtype=np.int64)
        map_probability = np.full(n_orbits, np.nan)
        entropy = np.full(n_orbits, np.nan)
        valid = np.all(np.isfinite(proba), axis=1)
        if np.any(valid):
            p = proba[valid]
            idx = np.argmax(p, axis=1)
            labels[valid] = self.class_labels[idx]
            map_probability[valid] = p[np.arange(len(p)), idx]
            p_safe = np.clip(p, 1e-300, 1.0)
            entropy[valid] = -np.sum(p_safe * np.log2(p_safe), axis=1)
        return labels, map_probability, entropy

    def save(self, path: str) -> str:
        """Write the fitted model to a compact ``.npz`` file.

        Parameters
        ----------
        path : str
            Destination path (``.npz`` appended by ``numpy`` if missing).

        Returns
        -------
        path : str
            The path written to.
        """
        np.savez(
            path,
            class_labels=self.class_labels,
            means=self.means,
            cholesky=self.cholesky,
            log_det_cov=self.log_det_cov,
            diagonal=self.diagonal,
            n_training=self.n_training,
            feature_names=np.array(self.feature_names),
            _format=np.array(self._format),
            _version=np.array(self._version),
        )
        return path if str(path).endswith(".npz") else f"{path}.npz"

    @classmethod
    def load(cls, path: str) -> "BayesianOrbitClassifier":
        """Load a model previously written by :meth:`save`.

        Parameters
        ----------
        path : str
            Path to the ``.npz`` file.

        Returns
        -------
        classifier : BayesianOrbitClassifier
            The reconstructed model.

        Raises
        ------
        ValueError
            If ``path`` was not written by :meth:`save`.
        """
        with np.load(path, allow_pickle=False) as f:
            if str(f["_format"]) != "lanfear.BayesianOrbitClassifier":
                raise ValueError(
                    f"{path} was not written by BayesianOrbitClassifier.save()."
                )
            return cls(
                class_labels=f["class_labels"],
                means=f["means"],
                cholesky=f["cholesky"],
                log_det_cov=f["log_det_cov"],
                diagonal=f["diagonal"],
                n_training=f["n_training"],
                feature_names=tuple(str(s) for s in f["feature_names"]),
                _version=int(f["_version"]),
            )


def fit_probabilistic_classifier(
    results,
    classification: Optional[OrbitClassification] = None,
    circ_thresh: float = 0.9,
    diffusion_threshold: Optional[float] = 0.1,
    inner_outer_ratio: float = 1.0,
    min_circ_margin: float = 0.05,
    min_diffusion_margin_dex: float = 0.3,
    min_xtube_margin_dex: float = 0.1,
    reg_covar: float = 1e-6,
    min_samples_full: Optional[int] = None,
    min_samples_diag: int = 5,
) -> BayesianOrbitClassifier:
    """Self-supervised training: fit a :class:`BayesianOrbitClassifier` from classify_orbits' own labels.

    Runs (or reuses) :func:`~lanfear.classify.classify_orbits`, keeps only the
    orbits confidently away from every continuous threshold it applies (see
    :func:`_confident_training_mask`), and fits one Gaussian per class on
    those. No new orbit integration and no independently labelled reference
    set is required -- the deterministic classifier's own validated output
    *is* the training data, restricted to its unambiguous interior.

    Parameters
    ----------
    results : lanfear.OrbitResults
        Orbits to train on. A larger, more diverse population gives a better
        (and more completely class-covering) fit; a few hundred to a few
        thousand successfully integrated orbits is normally enough.
    classification : lanfear.OrbitClassification, optional
        A classification already computed from ``results``. If omitted,
        ``results.classify(circ_thresh=circ_thresh,
        diffusion_threshold=diffusion_threshold,
        inner_outer_ratio=inner_outer_ratio)`` is run. Pass this explicitly if
        you classified with non-default parameters, so the margins below stay
        consistent with the thresholds actually used.
    circ_thresh, diffusion_threshold, inner_outer_ratio
        The same thresholds used to build ``classification`` (see
        :func:`~lanfear.classify.classify_orbits`) -- needed here to measure
        each orbit's margin from them, not to reclassify.
    min_circ_margin : float, optional
        Minimum ``|max(circulation) - circ_thresh|`` for a training orbit.
    min_diffusion_margin_dex : float, optional
        Minimum ``|log10(diffusion rate) - log10(diffusion_threshold)|`` (in
        decades) for a training orbit. Ignored if ``diffusion_threshold`` is
        ``None``.
    min_xtube_margin_dex : float, optional
        Minimum ``|log10(x_tube_ratio) - log10(inner_outer_ratio)|`` (in
        decades) for a long-axis-tube training orbit.
    reg_covar, min_samples_full, min_samples_diag
        Passed through to :meth:`BayesianOrbitClassifier.fit`.

    Returns
    -------
    model : BayesianOrbitClassifier
        The fitted model.

    Raises
    ------
    ValueError
        If ``classification`` does not match ``results``, or no orbit was
        confidently away from every threshold.
    """
    if classification is None:
        classification = results.classify(
            circ_thresh=circ_thresh,
            diffusion_threshold=diffusion_threshold,
            inner_outer_ratio=inner_outer_ratio,
        )
    else:
        _check_aligned(classification, results)

    features = feature_matrix(classification, results)
    confident = _confident_training_mask(
        classification,
        results,
        circ_thresh=circ_thresh,
        diffusion_threshold=diffusion_threshold,
        inner_outer_ratio=inner_outer_ratio,
        min_circ_margin=min_circ_margin,
        min_diffusion_margin_dex=min_diffusion_margin_dex,
        min_xtube_margin_dex=min_xtube_margin_dex,
    )
    n_confident = int(np.sum(confident))
    if n_confident == 0:
        raise ValueError(
            "No orbits were confidently away from every classification "
            "threshold; loosen the margins or supply a larger population."
        )
    logger.info(
        f"Training the Bayesian orbit classifier on {n_confident} of "
        f"{len(confident)} orbits confidently away from every classification "
        "threshold."
    )
    return BayesianOrbitClassifier.fit(
        features[confident],
        classification.labels[confident],
        reg_covar=reg_covar,
        min_samples_full=min_samples_full,
        min_samples_diag=min_samples_diag,
    )


@dataclass
class ProbabilisticOrbitClassification:
    """Per-orbit MAP label and posterior probability from a :class:`BayesianOrbitClassifier`.

    Produced by :func:`classify_orbits_probabilistic`. Deliberately does not
    store the full ``(n_orbits, n_classes)`` posterior matrix -- at scale that
    is the one part of this scheme worth being careful about memory-wise, and
    it is unnecessary: :attr:`features` (a few reduced numbers per orbit) and
    :attr:`model` (a few KB total) are enough to recompute any single orbit's
    full posterior on demand in microseconds, which is exactly what
    :meth:`probability_bar` and :meth:`probabilities_for` do. Pass a mask
    covering the whole population to :meth:`probabilities_for` if you
    genuinely need the full matrix.

    Parameters
    ----------
    labels : numpy.ndarray
        (n_orbits,) MAP :class:`~lanfear.OrbitClass` value, or ``-1`` where
        unevaluable (see :meth:`BayesianOrbitClassifier.predict`).
    map_probability : numpy.ndarray
        (n_orbits,) posterior probability of the MAP label.
    entropy : numpy.ndarray
        (n_orbits,) Shannon entropy of the posterior, in bits.
    ids : numpy.ndarray
        (n_orbits,) particle IDs, for :meth:`probability_bar`.
    model : BayesianOrbitClassifier
        The fitted model these labels/probabilities came from.
    features : numpy.ndarray
        (n_orbits, n_features) cached feature rows (see :func:`feature_matrix`),
        kept so a single orbit's posterior can be recomputed on demand.
    base : lanfear.OrbitClassification, optional
        The deterministic classification this was built from, if available
        (see :meth:`to_hard`).
    """

    labels: np.ndarray
    map_probability: np.ndarray
    entropy: np.ndarray
    ids: np.ndarray
    model: BayesianOrbitClassifier
    features: np.ndarray
    base: Optional[OrbitClassification] = None
    _id_sort_order: Optional[np.ndarray] = field(
        default=None, repr=False, compare=False
    )

    @property
    def class_names(self) -> Dict[int, str]:
        """Label-to-name mapping for the classes the model covers (see :attr:`model`)."""
        return self.model.class_names

    @property
    def names(self) -> np.ndarray:
        """Per-orbit class name strings (``"unevaluable"`` where :attr:`labels` is -1).

        Returns
        -------
        names : numpy.ndarray
            (n_orbits,) lower-case family names.
        """
        names = self.class_names
        return np.array([names.get(int(v), "unevaluable") for v in self.labels])

    def counts(self) -> dict:
        """Count the orbits assigned (by MAP) to each family.

        Returns
        -------
        counts : dict
            Mapping from class name to orbit count; includes
            ``"unevaluable"`` if any orbit had a non-finite feature.
        """
        names = self.class_names
        valid = self.labels >= 0
        vals, cnts = np.unique(self.labels[valid], return_counts=True)
        out = {names[int(v)]: int(c) for v, c in zip(vals, cnts)}
        n_bad = int(np.sum(~valid))
        if n_bad:
            out["unevaluable"] = n_bad
        return out

    def mask(self, cls) -> np.ndarray:
        """Boolean mask selecting orbits whose MAP label is ``cls``.

        Parameters
        ----------
        cls : OrbitClass or int
            The family to select.

        Returns
        -------
        mask : numpy.ndarray
            (n_orbits,) True where :attr:`labels` equals ``cls``.
        """
        return self.labels == int(cls)

    def mask_confident(self, threshold: float = 0.8) -> np.ndarray:
        """Orbits whose MAP posterior probability is at least ``threshold``.

        Parameters
        ----------
        threshold : float, optional
            Minimum :attr:`map_probability` to count as confident.

        Returns
        -------
        mask : numpy.ndarray
            (n_orbits,) True where confidently classified (False, not True,
            where unevaluable).
        """
        return np.isfinite(self.map_probability) & (self.map_probability >= threshold)

    def mask_ambiguous(self, threshold: float = 0.8) -> np.ndarray:
        """Orbits whose MAP posterior probability is below ``threshold``.

        The complement of :meth:`mask_confident` among *evaluable* orbits
        (unevaluable orbits are excluded from both, since ambiguity and
        unevaluability are different things).

        Parameters
        ----------
        threshold : float, optional
            Same threshold as :meth:`mask_confident`.

        Returns
        -------
        mask : numpy.ndarray
            (n_orbits,) True where evaluable but not confident.
        """
        return np.isfinite(self.map_probability) & (self.map_probability < threshold)

    def _index_for(self, id=None, index: Optional[int] = None) -> int:
        """Resolve a particle ID or a direct row index to a row index."""
        if (id is None) == (index is None):
            raise ValueError("pass exactly one of id or index.")
        if index is not None:
            return int(index)
        if self._id_sort_order is None:
            self._id_sort_order = np.argsort(self.ids)
        order = self._id_sort_order
        pos = np.searchsorted(self.ids[order], id)
        if pos >= len(order) or self.ids[order[pos]] != id:
            raise ValueError(f"particle id {id!r} not found.")
        return int(order[pos])

    def probability_bar(
        self, id=None, index: Optional[int] = None
    ) -> Tuple[list, np.ndarray]:
        """Posterior probability over every class for a single particle.

        Recomputed on demand from the cached feature row and the (tiny)
        fitted model -- no population-wide matrix is ever built.

        Parameters
        ----------
        id : optional
            Particle ID to look up (see :attr:`ids`).
        index : int, optional
            Row index to look up directly, instead of ``id``.

        Returns
        -------
        names : list of str
            Class names, in :attr:`model`'s :attr:`~BayesianOrbitClassifier.class_labels` order.
        probabilities : numpy.ndarray
            (n_classes,) posterior probability per class (NaN throughout if
            the particle's features were unevaluable).

        Raises
        ------
        ValueError
            If neither or both of ``id``/``index`` are given, or ``id`` is
            not found.
        """
        i = self._index_for(id=id, index=index)
        proba = self.model.predict_prob(self.features[i : i + 1])[0]
        names = [self.model.class_names[int(k)] for k in self.model.class_labels]
        return names, proba

    def probabilities_for(
        self, cls=None, mask: Optional[np.ndarray] = None
    ) -> Tuple[list, np.ndarray]:
        """Mean posterior probability over every class for a subset of orbits.

        E.g. "of the orbits currently MAP-labelled boxlet, what does the full
        posterior look like on average" -- a soft analogue of a single column
        of :meth:`~lanfear.OrbitClassification.compare`'s transition matrix.

        Parameters
        ----------
        cls : OrbitClass or int, optional
            Select orbits by their MAP label (shorthand for
            ``mask=self.mask(cls)``).
        mask : numpy.ndarray, optional
            Boolean (n_orbits,) mask selecting an arbitrary subset instead.

        Returns
        -------
        names : list of str
            Class names, in :attr:`model`'s class order.
        probabilities : numpy.ndarray
            (n_classes,) posterior probability per class, averaged over the
            selected orbits (unevaluable orbits are ignored in the average).

        Raises
        ------
        ValueError
            If neither or both of ``cls``/``mask`` are given, or the
            selection is empty.
        """
        if (cls is None) == (mask is None):
            raise ValueError("pass exactly one of cls or mask.")
        if mask is None:
            mask = self.mask(cls)
        if not np.any(mask):
            raise ValueError("no orbits match the given selection.")
        proba = self.model.predict_prob(self.features[mask])
        names = [self.model.class_names[int(k)] for k in self.model.class_labels]
        return names, np.nanmean(proba, axis=0)

    def plot_probability_bar(
        self,
        id=None,
        index: Optional[int] = None,
        cls=None,
        mask: Optional[np.ndarray] = None,
        ax=None,
        **kwargs,
    ):
        """Bar chart of the posterior probability over every class.

        Draws either a single particle's posterior (pass ``id`` or ``index``)
        or a subset's mean posterior (pass ``cls`` or ``mask``) -- exactly
        one selector should be given.

        Parameters
        ----------
        id, index
            Select a single particle; see :meth:`probability_bar`.
        cls, mask
            Select a subset; see :meth:`probabilities_for`.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. A new figure and axes are created if omitted.
        **kwargs
            Passed through to ``ax.bar``.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes the bars were drawn on.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        single = id is not None or index is not None
        if single:
            names, proba = self.probability_bar(id=id, index=index)
            title = f"particle {id}" if id is not None else f"orbit index {index}"
        else:
            names, proba = self.probabilities_for(cls=cls, mask=mask)
            title = "mean posterior"

        positions = np.arange(len(names))
        bar_kwargs = dict(kwargs)
        bar_kwargs.setdefault("color", [_colour_for(n, i) for i, n in enumerate(names)])
        bar_kwargs.setdefault("lw", 0.3)
        bar_kwargs.setdefault("ec", "k")
        ax.bar(positions, proba, **bar_kwargs)
        ax.set_xticks(positions)
        ax.set_xticklabels([_latex_label(n) for n in names], rotation=45, ha="right")
        ax.set_ylabel("posterior probability")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.figure.tight_layout()
        return ax

    def to_hard(self) -> OrbitClassification:
        """The deterministic :class:`~lanfear.OrbitClassification` with MAP labels swapped in.

        Lets the MAP labels from this Bayesian layer be fed straight into the
        existing plotting/comparison machinery on
        :class:`~lanfear.OrbitClassification` (``plot_class_fractions``,
        ``compare``, ...).

        Returns
        -------
        classification : lanfear.OrbitClassification
            A copy of :attr:`base` with ``labels`` replaced by this
            classification's MAP labels (unevaluable orbits become
            ``OrbitClass.UNCLASSIFIED``).

        Raises
        ------
        ValueError
            If :attr:`base` was not recorded (built without a ``results.classify()``-derived classification).
        """
        if self.base is None:
            raise ValueError(
                "no underlying OrbitClassification recorded; build this via "
                "classify_orbits_probabilistic() with its default classification."
            )
        labels = np.where(self.labels < 0, int(OrbitClass.UNCLASSIFIED), self.labels)
        return replace(self.base, labels=labels)


def classify_orbits_probabilistic(
    results,
    classification: Optional[OrbitClassification] = None,
    model: Optional[BayesianOrbitClassifier] = None,
    **fit_kwargs,
) -> ProbabilisticOrbitClassification:
    """Posterior-probability classification of the orbits in an :class:`~lanfear.OrbitResults`.

    Convenience entry point: classifies (if needed), fits (if no ``model`` is
    given) or reuses a :class:`BayesianOrbitClassifier`, and evaluates it on
    every orbit.

    Parameters
    ----------
    results : lanfear.OrbitResults
        Orbits to classify.
    classification : lanfear.OrbitClassification, optional
        Reused if given (and checked to match ``results``); otherwise
        ``results.classify()`` is run with any ``circ_thresh``/
        ``diffusion_threshold``/``inner_outer_ratio`` found in
        ``fit_kwargs``.
    model : BayesianOrbitClassifier, optional
        A previously fitted (and ideally saved/loaded, see
        :meth:`BayesianOrbitClassifier.save`) model to reuse. If omitted, one
        is fitted fresh via :func:`fit_probabilistic_classifier` -- fit this
        once on a representative population and pass it in on subsequent
        calls rather than refitting every run.
    **fit_kwargs
        Passed to :func:`fit_probabilistic_classifier` when ``model`` is not
        given.

    Returns
    -------
    classification : ProbabilisticOrbitClassification
        MAP label, posterior probability and entropy per orbit, plus the
        model and cached features needed for
        :meth:`~ProbabilisticOrbitClassification.probability_bar`.
    """
    if classification is None:
        classify_kwargs = {
            k: fit_kwargs[k]
            for k in ("circ_thresh", "diffusion_threshold", "inner_outer_ratio")
            if k in fit_kwargs
        }
        classification = results.classify(**classify_kwargs)
    else:
        _check_aligned(classification, results)

    if model is None:
        model = fit_probabilistic_classifier(
            results, classification=classification, **fit_kwargs
        )

    features = feature_matrix(classification, results)
    labels, map_probability, entropy = model.predict(features)
    return ProbabilisticOrbitClassification(
        labels=labels,
        map_probability=map_probability,
        entropy=entropy,
        ids=results.ids,
        model=model,
        features=features,
        base=classification,
    )
