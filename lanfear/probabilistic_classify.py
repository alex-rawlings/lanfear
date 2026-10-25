"""Measurement-error probabilities on top of :mod:`lanfear.classify`.

:func:`lanfear.classify.classify_orbits` labels each orbit by thresholding a
handful of diagnostics (circulation, hole size, frequency ratios, resonance
residuals, ...) measured over a finite integration window. Each diagnostic
carries a measurement error, and an orbit whose diagnostic lies within that
error of a threshold has a label that a slightly different measurement (a
longer window, other sampling) could change. This module turns those errors
into probabilities.

The error of each diagnostic is estimated from the orbit itself, by comparing
the two halves of its integration window (:func:`measurement_uncertainties`):
the angular-momentum circulation ``circ_a``, the closest approach to each axis
and the axis fundamental frequencies are all measured in both halves
(``L*_mean_h1``, ``rho_*_min_h1``/``_h2``, ``freq_*_h1``/``_h2``), and half
their difference is the error of the full-window value. The planar (azimuthal)
and radial frequencies, which have no half-window values, take the orbit's
largest axis-frequency error.

The errors of one orbit are correlated: a resonant orbit stays resonant in both
halves, so its frequencies differ between them along the resonance, not across
it. Each draw therefore moves all of an orbit's half-window diagnostics
together, by one Gaussian multiple of their half-differences; only the planar
and radial frequencies get independent deviates.

:func:`classify_orbits_probabilistic` draws ``n_draws`` such copies of every
orbit and classifies each copy with
the unchanged deterministic rules (and the same thresholds, including the
inner/outer cut placed on the original population), and records how often each
label comes out. The probability of a label is that fraction: **the chance
that the deterministic rule would give the label if the diagnostics were known
exactly**, given the measurement noise of this integration. Every label scheme
of :class:`~lanfear.OrbitClassification` gets one (classes, families, shapes,
boxlet types, sense, pattern resonances, regularity), with the correlations
between them kept, and the most probable label of each scheme is the
deterministic label except where more than half of the draws cross a
threshold.

What the probabilities do not include: regularity has no error model (the
spectral lines and the diffusion rate are not perturbed, so an orbit's
regularity is that of the deterministic classification, except for the rare
orbit whose spectral test responds to the perturbed frequency scale it uses as
a tolerance); errors that the two
halves of one window cannot reveal, such as how well the potential is fitted,
chaos that sets in after the window, or a mismatch between a rule's definition
and an orbit's true family, are not covered either. No model is fitted, so the
probabilities of an orbit do not depend on the rest of the population.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

import numpy as np

from ._logging import get_logger
from .classify import (
    BOXLET_TYPE_NAMES,
    CLASS_NAMES,
    PATTERN_RESONANCE_NAMES,
    REGULARITY_NAMES,
    SENSE_NAMES,
    SHAPE_NAMES,
    OrbitClassification,
    _colour_for,
    _latex_label,
    particle_weights,
)

logger = get_logger(__name__)

# Label schemes recorded per draw: scheme -> OrbitClassification attribute.
_SCHEME_ARRAYS = {
    "class": "labels",
    "family": "family",
    "shape": "shape",
    "boxlet": "boxlet_type",
    "sense": "sense",
    "pattern_resonance": "pattern_resonance",
    "regularity": "regularity",
}
_SCHEME_NAMES = {
    "class": CLASS_NAMES,
    "shape": SHAPE_NAMES,
    "boxlet": BOXLET_TYPE_NAMES,
    "sense": SENSE_NAMES,
    "pattern_resonance": PATTERN_RESONANCE_NAMES,
    "regularity": REGULARITY_NAMES,
}

# Axes whose leading line is weaker than this fraction of the strongest axis's
# do not set the orbit's frequency error (their frequency is noise-dominated).
_ACTIVE_AMP_FRAC = 0.05


def _require_columns(results, names) -> None:
    """Raise if ``results`` lacks any of the named summary columns."""
    missing = [n for n in names if n not in results.columns]
    if missing:
        raise ValueError(
            f"These orbit results lack the half-window columns {missing}; "
            f"re-run the orbit analysis with this version of lanfear"
        )


def measurement_uncertainties(results) -> Dict[str, np.ndarray]:
    """Per-orbit measurement errors of the diagnostics the classifier thresholds.

    Each diagnostic measured in both halves of the integration window gets half
    the (signed) difference second half minus first half: its magnitude is the
    error expected of the full-window value if the halves' errors are
    independent, and its direction (across the three axes) the way the
    diagnostics move together.

    Parameters
    ----------
    results : lanfear.OrbitResults
        Orbit results with the half-window summary columns and frequency
        data.

    Returns
    -------
    errors : dict
        ``"circulation"`` (N, 3): half-difference of ``circ_a = |<L_a>| /
        <|L_a|>``; ``"axis_distance"`` (N, 3): of the closest approach to each
        axis (``rho_*_min``, length units of the results); ``"frequency"``
        (N, 3): of each axis's leading frequency magnitude (HO angular
        frequency); ``"other_frequency"`` (N,): the largest frequency error
        among the oscillating axes, used for the planar and radial
        frequencies. Values that could not be measured are 0.

    Raises
    ------
    ValueError
        If the results have no half-window columns or no frequency data.
    """
    axes = "xyz"
    _require_columns(
        results,
        [f"L{a}_mean_h1" for a in axes]
        + [f"L{a}_abs_mean_h1" for a in axes]
        + [f"rho_{a}_min_h{h}" for a in axes for h in (1, 2)]
        + [f"freq_{a}_h{h}" for a in axes for h in (1, 2)],
    )
    if results.fundamentals is None:
        raise ValueError("These orbit results carry no frequency data")
    c = results.column
    n = len(results.ids)
    circulation = np.empty((n, 3))
    axis_distance = np.empty((n, 3))
    frequency = np.empty((n, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        for k, a in enumerate(axes):
            mean, abs_mean = c(f"L{a}_mean"), c(f"L{a}_abs_mean")
            mean_1, abs_mean_1 = c(f"L{a}_mean_h1"), c(f"L{a}_abs_mean_h1")
            # The halves hold equal numbers of samples, so the second half's
            # means follow from the full-window and first-half ones.
            mean_2, abs_mean_2 = 2.0 * mean - mean_1, 2.0 * abs_mean - abs_mean_1
            circ_1 = np.abs(mean_1) / np.maximum(abs_mean_1, 1e-30)
            circ_2 = np.abs(mean_2) / np.maximum(abs_mean_2, 1e-30)
            circulation[:, k] = 0.5 * (circ_2 - circ_1)
            axis_distance[:, k] = 0.5 * (c(f"rho_{a}_min_h2") - c(f"rho_{a}_min_h1"))
            frequency[:, k] = 0.5 * (
                np.abs(c(f"freq_{a}_h2")) - np.abs(c(f"freq_{a}_h1"))
            )
    frequency = np.nan_to_num(frequency, nan=0.0)
    amp = np.abs(np.asarray(results.lines[:, :, 0, 1], dtype=np.float64))
    active = amp >= _ACTIVE_AMP_FRAC * amp.max(axis=1, keepdims=True)
    other = np.max(np.where(active, np.abs(frequency), 0.0), axis=1)
    return {
        "circulation": np.nan_to_num(circulation, nan=0.0),
        "axis_distance": np.nan_to_num(axis_distance, nan=0.0),
        "frequency": frequency,
        "other_frequency": other,
    }


def _perturbed(results, errors, rng):
    """One draw of ``results`` with every thresholded diagnostic perturbed.

    The half-window diagnostics of an orbit move together, by one Gaussian
    multiple of their half-differences; the planar and radial frequencies get
    independent deviates.

    Parameters
    ----------
    results : lanfear.OrbitResults
        The measured orbits.
    errors : dict
        As returned by :func:`measurement_uncertainties`.
    rng : numpy.random.Generator
        Source of the Gaussian deviates.

    Returns
    -------
    draw : lanfear.OrbitResults
        A copy with perturbed circulation, axis distances and frequencies;
        the spectral lines and the diffusion are unchanged, so regularity is
        not perturbed.
    """
    n = len(results.ids)
    shared = rng.standard_normal(n)[:, None]  # one deviate per orbit
    columns = list(results.columns)
    summary = np.array(results.summary, dtype=np.float64, copy=True)
    for k, a in enumerate("xyz"):
        i_mean = columns.index(f"L{a}_mean")
        abs_mean = summary[:, columns.index(f"L{a}_abs_mean")]
        mean = summary[:, i_mean]
        circ = np.abs(mean) / np.maximum(abs_mean, 1e-30)
        circ = np.clip(circ + errors["circulation"][:, k] * shared[:, 0], 0.0, 1.0)
        sign = np.where(mean < 0, -1.0, 1.0)
        summary[:, i_mean] = sign * circ * abs_mean
        i_rho = columns.index(f"rho_{a}_min")
        summary[:, i_rho] = np.maximum(
            summary[:, i_rho] + errors["axis_distance"][:, k] * shared[:, 0], 0.0
        )
    other = errors["other_frequency"]
    if "radial_frequency" in columns:
        i_radial = columns.index("radial_frequency")
        summary[:, i_radial] = np.abs(
            summary[:, i_radial] + other * rng.standard_normal(n)
        )
    fundamentals = np.asarray(results.fundamentals, dtype=np.float64)
    magnitude = np.maximum(np.abs(fundamentals) + errors["frequency"] * shared, 0.0)
    fundamentals = np.where(fundamentals < 0, -1.0, 1.0) * magnitude
    planar_lines = results.planar_lines
    if planar_lines is not None:
        planar_lines = np.array(planar_lines, dtype=np.float64, copy=True)
        # One shift per plane: the planar lines move together.
        planar_lines[..., 0] += (other[:, None] * rng.standard_normal((n, 3)))[
            :, :, None
        ]
    return replace(
        results,
        summary=summary,
        fundamentals=fundamentals,
        planar_lines=planar_lines,
    )


@dataclass
class ProbabilisticOrbitClassification:
    """Label probabilities of each orbit under its measurement errors.

    Produced by :func:`classify_orbits_probabilistic`. For every label scheme
    (``"class"``, ``"family"``, ``"shape"``, ``"boxlet"``, ``"sense"``,
    ``"pattern_resonance"``, ``"regularity"``) it records how many of the
    :attr:`n_draws` perturbed copies of each orbit got each label; a label's
    probability is that count over :attr:`n_draws`.

    Parameters
    ----------
    ids : numpy.ndarray
        (N,) particle IDs.
    base : lanfear.OrbitClassification
        The deterministic classification of the measured orbits.
    n_draws : int
        Number of perturbed copies classified.
    seed : int
        Seed of the random draws.
    label_values : dict
        Scheme -> (K,) label values, the columns of :attr:`label_counts`.
    label_counts : dict
        Scheme -> (N, K) number of draws giving each label.
    names : dict
        Scheme -> {label value: name}.
    """

    ids: np.ndarray
    base: OrbitClassification
    n_draws: int
    seed: int
    label_values: Dict[str, np.ndarray]
    label_counts: Dict[str, np.ndarray]
    names: Dict[str, Dict[int, str]]
    _id_sort_order: Optional[np.ndarray] = field(
        default=None, repr=False, compare=False
    )

    @property
    def schemes(self) -> Tuple[str, ...]:
        """The label schemes with probabilities."""
        return tuple(self.label_counts)

    def _check_scheme(self, scheme: str) -> None:
        if scheme not in self.label_counts:
            raise ValueError(
                f"scheme must be one of {sorted(self.label_counts)}, got {scheme!r}"
            )

    def probabilities(self, scheme: str = "class") -> Tuple[list, np.ndarray]:
        """Probability of every label of a scheme, for every orbit.

        Parameters
        ----------
        scheme : str, optional
            Label scheme (see :attr:`schemes`; default ``"class"``).

        Returns
        -------
        names : list of str
            Label names, in column order.
        probabilities : numpy.ndarray
            (N, K) probabilities, each row summing to 1.
        """
        self._check_scheme(scheme)
        names = [self.names[scheme][int(v)] for v in self.label_values[scheme]]
        return names, self.label_counts[scheme] / float(self.n_draws)

    def map_labels(self, scheme: str = "class") -> np.ndarray:
        """Most probable label of each orbit.

        Ties go to the deterministic label.

        Parameters
        ----------
        scheme : str, optional
            Label scheme (default ``"class"``).

        Returns
        -------
        labels : numpy.ndarray
            (N,) label values.
        """
        self._check_scheme(scheme)
        counts = self.label_counts[scheme].astype(np.int64)
        values = self.label_values[scheme]
        deterministic = np.asarray(getattr(self.base, _SCHEME_ARRAYS[scheme]))
        own = values[None, :] == deterministic[:, None]
        # A half-count bonus breaks ties in favour of the deterministic label.
        k = np.argmax(2 * counts + own, axis=1)
        return values[k]

    def map_probability(self, scheme: str = "class") -> np.ndarray:
        """Probability of each orbit's most probable label.

        Parameters
        ----------
        scheme : str, optional
            Label scheme (default ``"class"``).

        Returns
        -------
        probability : numpy.ndarray
            (N,) values in ``[1 / K, 1]``.
        """
        self._check_scheme(scheme)
        return self.label_counts[scheme].max(axis=1) / float(self.n_draws)

    def entropy(self, scheme: str = "class") -> np.ndarray:
        """Shannon entropy of each orbit's label probabilities, in bits.

        Parameters
        ----------
        scheme : str, optional
            Label scheme (default ``"class"``).

        Returns
        -------
        entropy : numpy.ndarray
            (N,) 0 for a label every draw agrees on.
        """
        _, p = self.probabilities(scheme)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p > 0, -p * np.log2(p), 0.0)
        return terms.sum(axis=1)

    def map_names(self, scheme: str = "class") -> np.ndarray:
        """Names of the most probable labels.

        Parameters
        ----------
        scheme : str, optional
            Label scheme (default ``"class"``).

        Returns
        -------
        names : numpy.ndarray
            (N,) strings.
        """
        names = self.names[scheme]
        return np.array([names[int(v)] for v in self.map_labels(scheme)])

    def counts(self, scheme: str = "class", particles=None) -> dict:
        """Expected number of orbits (or mass) with each label.

        The sum of the label probabilities over the orbits, so an orbit split
        between two labels contributes to both.

        Parameters
        ----------
        scheme : str, optional
            Label scheme (default ``"class"``).
        particles : lanfear.ParticleSystem, optional
            If given, weight each orbit by its particle's mass (matched by
            ID, :func:`lanfear.classify.particle_weights`).

        Returns
        -------
        counts : dict
            Label name -> expected count (or mass), for labels with a
            non-zero expectation.
        """
        names, p = self.probabilities(scheme)
        weights = particle_weights(self.ids, particles)
        totals = weights @ p
        return {n: float(t) for n, t in zip(names, totals) if t > 0}

    def counts_uncertainty(self, scheme: str = "class", particles=None) -> dict:
        """Measurement-error spread of :meth:`counts`.

        Treating the orbits as independent, the standard deviation of the
        count of a label is ``sqrt(sum_i w_i^2 p_i (1 - p_i))``. It adds to,
        and is independent of, the sampling error of a finite population
        (see :meth:`lanfear.OrbitClassification.fractions_by`).

        Parameters
        ----------
        scheme : str, optional
            Label scheme (default ``"class"``).
        particles : lanfear.ParticleSystem, optional
            If given, weight each orbit by its particle's mass.

        Returns
        -------
        sigma : dict
            Label name -> standard deviation of the expected count (or mass),
            for the labels of :meth:`counts`.
        """
        names, p = self.probabilities(scheme)
        weights = particle_weights(self.ids, particles)
        variance = (weights**2) @ (p * (1.0 - p))
        totals = weights @ p
        return {
            n: float(np.sqrt(v)) for n, v, t in zip(names, variance, totals) if t > 0
        }

    def mask(self, label, scheme: str = "class") -> np.ndarray:
        """Orbits whose most probable label is ``label``.

        Parameters
        ----------
        label : int or str
            Label value or name.
        scheme : str, optional
            Label scheme (default ``"class"``).

        Returns
        -------
        mask : numpy.ndarray
            (N,) boolean.
        """
        if isinstance(label, str):
            return self.map_names(scheme) == label
        return self.map_labels(scheme) == int(label)

    def mask_confident(
        self, threshold: float = 0.8, scheme: str = "class"
    ) -> np.ndarray:
        """Orbits whose most probable label has at least ``threshold`` probability.

        Parameters
        ----------
        threshold : float, optional
            Minimum probability (default 0.8).
        scheme : str, optional
            Label scheme (default ``"class"``).

        Returns
        -------
        mask : numpy.ndarray
            (N,) boolean.
        """
        return self.map_probability(scheme) >= threshold

    def mask_ambiguous(
        self, threshold: float = 0.8, scheme: str = "class"
    ) -> np.ndarray:
        """Orbits whose most probable label has less than ``threshold`` probability.

        Parameters
        ----------
        threshold : float, optional
            Same threshold as :meth:`mask_confident` (default 0.8).
        scheme : str, optional
            Label scheme (default ``"class"``).

        Returns
        -------
        mask : numpy.ndarray
            (N,) boolean, the complement of :meth:`mask_confident`.
        """
        return ~self.mask_confident(threshold, scheme)

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
        self, id=None, index: Optional[int] = None, scheme: str = "class"
    ) -> Tuple[list, np.ndarray]:
        """Label probabilities of a single orbit.

        Parameters
        ----------
        id : optional
            Particle ID to look up (see :attr:`ids`).
        index : int, optional
            Row index to look up directly, instead of ``id``.
        scheme : str, optional
            Label scheme (default ``"class"``).

        Returns
        -------
        names : list of str
            Label names.
        probabilities : numpy.ndarray
            (K,) probability of each label.

        Raises
        ------
        ValueError
            If neither or both of ``id``/``index`` are given, or ``id`` is
            not found.
        """
        i = self._index_for(id=id, index=index)
        names, p = self.probabilities(scheme)
        return names, p[i]

    def probabilities_for(
        self, label=None, mask: Optional[np.ndarray] = None, scheme: str = "class"
    ) -> Tuple[list, np.ndarray]:
        """Mean label probabilities of a subset of orbits.

        Parameters
        ----------
        label : int or str, optional
            Select the orbits whose most probable label is this (shorthand for
            ``mask=self.mask(label, scheme)``).
        mask : numpy.ndarray, optional
            Boolean (N,) mask selecting an arbitrary subset instead.
        scheme : str, optional
            Label scheme (default ``"class"``).

        Returns
        -------
        names : list of str
            Label names.
        probabilities : numpy.ndarray
            (K,) mean probability of each label over the subset.

        Raises
        ------
        ValueError
            If neither or both of ``label``/``mask`` are given, or the
            selection is empty.
        """
        if (label is None) == (mask is None):
            raise ValueError("pass exactly one of label or mask.")
        if mask is None:
            mask = self.mask(label, scheme)
        if not np.any(mask):
            raise ValueError("no orbits match the given selection.")
        names, p = self.probabilities(scheme)
        return names, p[mask].mean(axis=0)

    def plot_probability_bar(
        self,
        id=None,
        index: Optional[int] = None,
        label=None,
        mask: Optional[np.ndarray] = None,
        scheme: str = "class",
        ax=None,
        **kwargs,
    ):
        """Bar chart of the label probabilities of one orbit or of a subset.

        Pass ``id`` or ``index`` for a single orbit (:meth:`probability_bar`),
        or ``label`` or ``mask`` for a subset's mean
        (:meth:`probabilities_for`). Labels no draw gave are omitted.

        Parameters
        ----------
        id, index
            Select a single orbit.
        label, mask
            Select a subset.
        scheme : str, optional
            Label scheme (default ``"class"``).
        ax : matplotlib.axes.Axes, optional
            Axes to draw into; a new figure is created if omitted.
        **kwargs
            Passed to ``ax.bar``.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes drawn on.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        if id is not None or index is not None:
            names, p = self.probability_bar(id=id, index=index, scheme=scheme)
            title = f"particle {id}" if id is not None else f"orbit index {index}"
        else:
            names, p = self.probabilities_for(label=label, mask=mask, scheme=scheme)
            title = "mean probability"
        shown = np.flatnonzero(p > 0)
        names = [names[k] for k in shown]
        p = p[shown]
        positions = np.arange(len(names))
        bar_kwargs = dict(kwargs)
        bar_kwargs.setdefault("color", [_colour_for(n, i) for i, n in enumerate(names)])
        bar_kwargs.setdefault("lw", 0.3)
        bar_kwargs.setdefault("ec", "k")
        ax.bar(positions, p, **bar_kwargs)
        ax.set_xticks(positions)
        ax.set_xticklabels([_latex_label(n) for n in names], rotation=45, ha="right")
        ax.set_ylabel("probability")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.figure.tight_layout()
        return ax

    def to_hard(self) -> OrbitClassification:
        """The deterministic classification with the most probable labels swapped in.

        Every scheme's label array is replaced by its most probable label, so
        the plotting and comparison machinery of
        :class:`~lanfear.OrbitClassification` can be used on them.

        Returns
        -------
        classification : lanfear.OrbitClassification
            A copy of :attr:`base`.
        """
        updates = {
            attr: self.map_labels(scheme)
            for scheme, attr in _SCHEME_ARRAYS.items()
            if scheme in self.label_counts
        }
        return replace(self.base, **updates)


def classify_orbits_probabilistic(
    results,
    classification: Optional[OrbitClassification] = None,
    n_draws: int = 32,
    seed: int = 0,
    drop_binary_interacting: bool = False,
    drop_runaways: bool = True,
    **classify_kwargs,
) -> ProbabilisticOrbitClassification:
    """Label probabilities of the orbits under their measurement errors.

    Draws ``n_draws`` copies of every orbit's diagnostics within their
    measurement errors (:func:`measurement_uncertainties`), classifies each
    copy with :func:`~lanfear.classify.classify_orbits` and the same settings
    as the deterministic classification, and counts the labels (see the module
    notes for what the probabilities mean and leave out).

    Parameters
    ----------
    results : lanfear.OrbitResults
        Orbits to classify, with the half-window summary columns and
        frequency data.
    classification : lanfear.OrbitClassification, optional
        The deterministic classification of ``results`` (after the drops
        below); computed with ``classify_kwargs`` if omitted. Its
        inner/outer cut is reused for every draw.
    n_draws : int, optional
        Number of perturbed copies (default 32; probabilities are multiples
        of ``1 / n_draws``). Each costs one classification of the
        population.
    seed : int, optional
        Seed of the random draws (default 0).
    drop_binary_interacting : bool, optional
        Remove the orbits flagged by :attr:`OrbitResults.binary_interacting`
        first (default False).
    drop_runaways : bool, optional
        Remove the orbits flagged by :attr:`OrbitResults.runaway` first
        (default True, as :func:`~lanfear.classify.classify_orbits`).
    **classify_kwargs
        Passed to :func:`~lanfear.classify.classify_orbits` for the
        deterministic classification and every draw. Use the same values as
        for a ``classification`` passed in.

    Returns
    -------
    classification : ProbabilisticOrbitClassification
        The label counts of every scheme.

    Raises
    ------
    ValueError
        If ``n_draws`` is not in ``[1, 65535]``, the results lack the
        half-window columns or frequency data, or ``classification`` does
        not match ``results``.
    """
    if not 1 <= n_draws <= np.iinfo(np.uint16).max:
        raise ValueError(f"n_draws must be in [1, 65535], got {n_draws}")
    if drop_binary_interacting:
        results = results.drop_binary_interacting()
    if drop_runaways:
        results = results.drop_runaways()
    classify_kwargs = dict(classify_kwargs, drop_runaways=False)
    if classification is None:
        classification = results.classify(**classify_kwargs)
    elif len(classification.labels) != len(results.ids) or (
        classification.ids is not None
        and not np.array_equal(classification.ids, results.ids)
    ):
        raise ValueError(
            "classification does not match results (were the same orbits "
            "dropped from both?)"
        )
    errors = measurement_uncertainties(results)
    classify_kwargs["inner_outer_ratio"] = classification.inner_outer_ratio

    names = dict(_SCHEME_NAMES)
    names["family"] = dict(classification.family_names)
    values = {s: np.array(sorted(n), dtype=np.int64) for s, n in names.items()}
    dtype = np.uint8 if n_draws <= np.iinfo(np.uint8).max else np.uint16
    n = len(results.ids)
    counts = {s: np.zeros((n, len(values[s])), dtype=dtype) for s in values}
    rows = np.arange(n)

    rng = np.random.default_rng(seed)
    package_logger = logging.getLogger("lanfear")
    level = package_logger.level
    package_logger.setLevel(logging.ERROR)  # per-draw summaries would repeat
    try:
        for _ in range(n_draws):
            draw = _perturbed(results, errors, rng).classify(**classify_kwargs)
            for scheme, attr in _SCHEME_ARRAYS.items():
                labels = np.asarray(getattr(draw, attr), dtype=np.int64)
                # values[scheme] is sorted and holds every label value.
                index = np.searchsorted(values[scheme], labels)
                counts[scheme][rows, index] += 1
    finally:
        package_logger.setLevel(level)

    out = ProbabilisticOrbitClassification(
        ids=np.asarray(results.ids),
        base=classification,
        n_draws=n_draws,
        seed=seed,
        label_values=values,
        label_counts=counts,
        names=names,
    )
    changed = np.mean(out.map_labels("class") != classification.labels)
    certain = np.mean(out.map_probability("family") == 1.0)
    logger.info(
        f"Classified {n_draws} perturbed copies of {n} orbits: {100 * certain:.1f}% "
        f"keep their family in every draw; the most probable class differs from "
        f"the deterministic one for {100 * changed:.2f}%"
    )
    return out
