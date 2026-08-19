"""Orbit classification from integrated + frequency-analysed orbits.

Following the standard picture (e.g. Carpintero & Aguilar 1998; Frigo et al.
2021), orbits are assigned to families from three cheap, already-reduced
per-orbit quantities:

* **Angular-momentum circulation.** A tube (loop) orbit circulates about one
  principal axis, so the corresponding component of angular momentum keeps its
  sign: ``circ_a = |<L_a>| / <|L_a|>`` is near 1. A box circulates about none
  (all ``circ_a`` small). The circulating axis names the tube: z -> short-axis
  tube, x -> long-axis tube, y -> (unstable) intermediate-axis tube.
* **Shape tensor.** Its smallest eigenvalue vanishes for a planar orbit in any
  orientation, identifying rosettes (2-D loops, typical of near-spherical
  potentials) and separating them from thick 3-D tubes.
* **Fundamental frequencies.** A low-order commensurability
  ``n_x w_x + n_y w_y + n_z w_z ~ 0`` among the axis fundamentals marks a
  resonant box (boxlet). Requires frequency data (:func:`analyse_family`).
* **Number of base frequencies.** An orbit whose spectrum needs more than three
  independent base frequencies is *irregular* (Frigo et al. 2021) -- it is not
  confined to a regular 3-torus and is a likely-chaotic candidate. This label
  overrides the regular-family assignment and requires frequency data.

The classification runs on the compact arrays in an :class:`OrbitResults`, so it
is trivially fast even for millions of orbits.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import IntEnum
from math import gcd
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt

from ._logging import get_logger

logger = get_logger(__name__)


class OrbitClass(IntEnum):
    """Enumeration of orbit families assigned by :func:`classify_orbits`."""

    UNCLASSIFIED = 0
    PIBOX = 1
    BOXLET = 2  # resonant box (banana/fish/pretzel/...)
    INTERMEDIATE_AXIS_TUBE = 3  # y-tube (generally unstable)
    SHORT_AXIS_TUBE = 4  # z-tube
    INNER_LONG_AXIS_TUBE = 5  # inner x-tube
    OUTER_LONG_AXIS_TUBE = 6  # outer x-tube
    ROSETTE = 7  # planar loop
    IRREGULAR = 8  # > 3 base frequencies (Frigo et al. 2021) -- likely chaotic


CLASS_NAMES = {c.value: c.name.lower() for c in OrbitClass}


class OrbitFamily(IntEnum):
    """Condensed orbit families (box vs tube).

    Produced by :meth:`OrbitClassification.condense_families`, which groups the
    detailed :class:`OrbitClass` subclasses into the fundamental box/tube
    dichotomy.
    """

    UNCLASSIFIED = 0
    BOX = 1
    TUBE = 2


CONDENSED_NAMES = {f.value: f.name.lower() for f in OrbitFamily}

# Which detailed OrbitClass subclasses fold into each condensed OrbitFamily.
_TUBE_CLASSES = (
    OrbitClass.SHORT_AXIS_TUBE,
    OrbitClass.INNER_LONG_AXIS_TUBE,
    OrbitClass.OUTER_LONG_AXIS_TUBE,
    OrbitClass.INTERMEDIATE_AXIS_TUBE,
    OrbitClass.ROSETTE,  # a rosette is a (planar) loop orbit
)
_BOX_CLASSES = (OrbitClass.PIBOX, OrbitClass.BOXLET)

# LaTeX labels used when rendering family names on plot legends and axes (see
# the OrbitClassification.plot_* methods). Keyed by the family name as it appears
# in CLASS_NAMES / CONDENSED_NAMES; values are raw strings holding the LaTeX to
# render (e.g. r"$x$-tube"). FILL THESE IN -- an empty entry falls back to the
# plain family name via _latex_label(), so plots stay legible until you do.
LATEX_LABELS = {
    "unclassified": r"$\mathrm{unclassified}$",
    "pibox": r"$\pi\mathrm{-box}$",
    "boxlet": r"$\mathrm{boxlet}$",
    "intermediate_axis_tube": r"$y\mathrm{-tube}$",
    "short_axis_tube": r"$z\mathrm{-tube}$",
    "inner_long_axis_tube": r"$\mathrm{inner}\;x\mathrm{-tube}$",
    "outer_long_axis_tube": r"$\mathrm{outer}\;x\mathrm{-tube}$",
    "rosette": r"$\mathrm{rosette}$",
    "irregular": r"$\mathrm{irregular}$",
    "tube": r"$\mathrm{tube}$",  # (condensed family)
    "box": r"$\mathrm{box}$",  # (condensed family)
}


def _latex_label(name: str) -> str:
    """LaTeX label for a family name, falling back to the plain name.

    Parameters
    ----------
    name : str
        Family name as stored in :data:`CLASS_NAMES` / :data:`CONDENSED_NAMES`.

    Returns
    -------
    label : str
        The corresponding entry in :data:`LATEX_LABELS` if it is non-empty,
        otherwise ``name`` unchanged.
    """
    return LATEX_LABELS.get(name) or name


# Default colour palette used for orbit-class plots when no other palette is
# requested. Keyed by the family name as it appears in CLASS_NAMES (mirrors
# LATEX_LABELS above).
DEFAULT_PALETTE = {
    "unclassified": "#999999",
    "pibox": "#F4477E",
    "boxlet": "#FF8552",
    "intermediate_axis_tube": "#FFC145",  # y-tube
    "short_axis_tube": "#21B0A6",  # z-tube
    "inner_long_axis_tube": "#A64CA6",  # inner x-tube
    "outer_long_axis_tube": "#6F4E9C",  # outer x-tube
    "rosette": "#182B54",
    "irregular": "#797878FF",
    "tube": "#D98FB2",  # condensed family
    "box": "#141A3A",  # condensed family
}


def _colour_for(name: str, index: int = 0) -> str:
    """Colour for a family name from :data:`DEFAULT_PALETTE`.

    Parameters
    ----------
    name : str
        Family name as stored in :data:`CLASS_NAMES` / :data:`CONDENSED_NAMES`.
    index : int, optional
        Position of ``name`` among the families being plotted, used to pick a
        colour from matplotlib's default cycle when ``name`` has no entry in
        :data:`DEFAULT_PALETTE` (e.g. the condensed "box"/"tube" families).

    Returns
    -------
    colour : str
        A matplotlib-compatible colour spec.
    """
    if name in DEFAULT_PALETTE:
        return DEFAULT_PALETTE[name]
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    return cycle[index % len(cycle)]


@dataclass
class OrbitClassification:
    """Result of :func:`classify_orbits` (all arrays indexed like the orbits).

    Parameters
    ----------
    labels : numpy.ndarray
        (N,) :class:`OrbitClass` integer values.
    circulation : numpy.ndarray
        (N, 3) ``|<L_a>| / <|L_a|>`` per axis.
    tube_axis : numpy.ndarray
        (N,) index (0=x, 1=y, 2=z) of the dominant circulation axis.
    planarity : numpy.ndarray
        (N,) ``lambda_min / lambda_max`` of the shape tensor.
    resonance : numpy.ndarray
        (N, 3) primitive resonance vector (all zero if none found).
    resonance_order : numpy.ndarray
        (N,) L1 order ``|n|_1`` of the resonance (0 if none found).
    class_names : dict, optional
        Mapping from integer label to name, used by :attr:`names` and
        :meth:`counts`. Defaults to the detailed :data:`CLASS_NAMES`; a condensed
        result (from :meth:`condense_families`) carries :data:`CONDENSED_NAMES`.
    radius : numpy.ndarray, optional
        (N,) characteristic radius of each orbit, in *physical* length units.
        This is the instantaneous snapshot radius (the orbit's radius at
        integration start, measured from the galaxy centre) and is the default
        binning radius for :meth:`plot_class_fractions`. Recorded by
        :func:`classify_orbits`; ``None`` when no radius is available.
    radius_orbit_averaged : numpy.ndarray, optional
        (N,) time-averaged radius ``r_mean`` of each orbit, in *physical* length
        units. Pass this to :meth:`plot_class_fractions` as ``radius=`` to bin
        on the orbit-averaged radius instead of the snapshot radius. Recorded by
        :func:`classify_orbits`; ``None`` when not available.
    ids : numpy.ndarray, optional
        (N,) particle ID of each orbit, recorded by :func:`classify_orbits` and
        used by :meth:`compare` to match particles between two classifications.
        ``None`` when not available.
    fundamentals : numpy.ndarray, optional
        (N, 3) signed fundamental frequency per axis, recorded by
        :func:`classify_orbits` when frequency data is available and used by
        :meth:`plot_frequency_map`. ``None`` when not available.
    """

    labels: np.ndarray  # (N,) OrbitClass values
    circulation: np.ndarray  # (N,3) |<L_a>|/<|L_a|> per axis
    tube_axis: np.ndarray  # (N,) index of dominant circulation axis
    planarity: np.ndarray  # (N,) lambda_min/lambda_max of shape tensor
    resonance: np.ndarray  # (N,3) primitive resonance vector (0 if none)
    resonance_order: np.ndarray  # (N,) |n|_1 of the resonance (0 if none)
    class_names: Dict[int, str] = field(default_factory=lambda: dict(CLASS_NAMES))
    radius: Optional[np.ndarray] = None  # (N,) snapshot radius, physical units
    radius_orbit_averaged: Optional[np.ndarray] = None  # (N,) r_mean, physical
    ids: Optional[np.ndarray] = None  # (N,) particle IDs
    fundamentals: Optional[np.ndarray] = None  # (N,3) signed fund. freq per axis

    @property
    def names(self) -> np.ndarray:
        """Per-orbit class name strings.

        Returns
        -------
        names : numpy.ndarray
            (N,) lower-case family names (e.g. ``"short_axis_tube"``, or
            ``"tube"``/``"box"`` for a condensed result).
        """
        return np.array([self.class_names[int(v)] for v in self.labels])

    def counts(self) -> dict:
        """Count the orbits in each family.

        Returns
        -------
        counts : dict
            Mapping from class name to the number of orbits in that family.
        """
        vals, cnts = np.unique(self.labels, return_counts=True)
        return {self.class_names[int(v)]: int(c) for v, c in zip(vals, cnts)}

    def mask(self, cls) -> np.ndarray:
        """Boolean mask selecting orbits of a given family.

        Parameters
        ----------
        cls : OrbitClass or OrbitFamily or int
            The family to select (e.g. ``OrbitClass.SHORT_AXIS_TUBE`` on a full
            result, or ``OrbitFamily.TUBE`` on a condensed one).

        Returns
        -------
        mask : numpy.ndarray
            (N,) True where the orbit belongs to ``cls``.
        """
        return self.labels == int(cls)

    def get_class_ids(self, cls) -> np.ndarray:
        """Particle IDs of the orbits belonging to a given family.

        Parameters
        ----------
        cls : OrbitClass or OrbitFamily or int
            The family to select (e.g. ``OrbitClass.SHORT_AXIS_TUBE`` on a full
            result, or ``OrbitFamily.TUBE`` on a condensed one).

        Returns
        -------
        ids : numpy.ndarray
            Particle IDs of the orbits belonging to ``cls``.

        Raises
        ------
        ValueError
            If this classification carries no particle IDs.
        """
        if self.ids is None:
            raise ValueError(
                "no particle IDs available; build the classification with "
                "classify_orbits (which records them from OrbitResults.ids)."
            )
        return self.ids[self.mask(cls)]

    def condense_families(self) -> "OrbitClassification":
        """Group the orbit subclasses into the box/tube dichotomy.

        All tube subclasses (short-axis, inner/outer long-axis and
        intermediate-axis tubes, and rosettes) fold into
        :attr:`OrbitFamily.TUBE`; boxes and boxlets fold into
        :attr:`OrbitFamily.BOX`; unclassified orbits stay unclassified. The
        per-orbit diagnostic arrays (circulation, tube axis, planarity,
        resonance) are carried through unchanged.

        Returns
        -------
        condensed : OrbitClassification
            A new classification whose ``labels`` are :class:`OrbitFamily`
            values and whose :attr:`names`/:meth:`counts` report ``"box"``,
            ``"tube"`` or ``"unclassified"``.
        """
        if self.class_names == CONDENSED_NAMES:
            # Already condensed -- return an equivalent copy (idempotent). The
            # OrbitClass -> family map cannot be reapplied to family labels.
            return OrbitClassification(
                labels=self.labels.copy(),
                circulation=self.circulation,
                tube_axis=self.tube_axis,
                planarity=self.planarity,
                resonance=self.resonance,
                resonance_order=self.resonance_order,
                class_names=dict(CONDENSED_NAMES),
                radius=self.radius,
                radius_orbit_averaged=self.radius_orbit_averaged,
                ids=self.ids,
                fundamentals=self.fundamentals,
            )
        condensed = np.full(self.labels.shape, OrbitFamily.UNCLASSIFIED, dtype=np.int64)
        for cls in _TUBE_CLASSES:
            condensed[self.labels == int(cls)] = OrbitFamily.TUBE
        for cls in _BOX_CLASSES:
            condensed[self.labels == int(cls)] = OrbitFamily.BOX
        return OrbitClassification(
            labels=condensed,
            circulation=self.circulation,
            tube_axis=self.tube_axis,
            planarity=self.planarity,
            resonance=self.resonance,
            resonance_order=self.resonance_order,
            class_names=dict(CONDENSED_NAMES),
            radius=self.radius,
            radius_orbit_averaged=self.radius_orbit_averaged,
            ids=self.ids,
            fundamentals=self.fundamentals,
        )

    def plot_class_fractions(
        self, edges, per_bin: bool = True, radius=None, ax=None, **kwargs
    ):
        """Plot the relative frequency of each orbit class in radial bins.

        Orbits are binned by their characteristic radius, and the fraction of
        orbits belonging to each class is drawn as a curve against radius (one
        line per class present).

        Parameters
        ----------
        edges : array_like
            (n_bins + 1,) monotonically increasing radial bin edges, in
            *physical* length units (matching :attr:`radius`).
        per_bin : bool, optional
            Normalisation of the frequencies. If ``True`` (default), each
            class count in a bin is divided by the number of orbits in that
            bin, so the curves give the class composition within each bin
            (summing to 1 across classes). If ``False``, counts are divided by
            the total number of binned orbits, so the curves give each class's
            share of the whole population.
        radius : array_like, optional
            (N,) per-orbit radius to bin on, in *physical* length units.
            Defaults to :attr:`radius` (the instantaneous snapshot radius). Pass
            :attr:`radius_orbit_averaged` to bin on the orbit-averaged radius
            instead, or any other per-orbit physical radius of your own.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. A new figure and axes are created if omitted.
        **kwargs
            Passed through to ``ax.plot`` for every class. Pass ``color=...``
            to override :data:`DEFAULT_PALETTE` for all classes at once.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes the curves were drawn on.

        Raises
        ------
        ValueError
            If no radius is available, or ``radius``/``edges`` are malformed.
        ImportError
            If matplotlib is not installed.
        """
        r = self.radius if radius is None else radius
        if r is None:
            raise ValueError(
                "no per-orbit radius available; pass radius=... or build the "
                "classification with classify_orbits (which records the "
                "snapshot radius)."
            )
        r = np.asarray(r, dtype=float)
        if r.shape != self.labels.shape:
            raise ValueError("radius and labels must have the same length.")

        edges = np.asarray(edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("edges must be a 1-D array of at least two bin edges.")
        n_bins = edges.size - 1
        centres = 0.5 * (edges[:-1] + edges[1:])

        bin_index = np.digitize(r, edges) - 1  # 0..n_bins-1 within range
        in_range = (bin_index >= 0) & (bin_index < n_bins)

        bin_total = np.bincount(bin_index[in_range], minlength=n_bins).astype(float)
        grand_total = float(in_range.sum())
        if grand_total == 0:
            logger.warning("No orbits fall within the given radial edges.")

        if ax is None:
            _, ax = plt.subplots()

        for i, cls in enumerate(sorted(int(v) for v in np.unique(self.labels))):
            selected = in_range & (self.labels == cls)
            count = np.bincount(bin_index[selected], minlength=n_bins).astype(float)
            if per_bin:
                normalisation = bin_total.copy()
                # Empty bins have no defined composition -> leave a gap (NaN).
                normalisation[normalisation == 0] = np.nan
                frequency = count / normalisation
            else:
                frequency = count / grand_total if grand_total > 0 else count
            plot_kwargs = dict(kwargs)
            plot_kwargs.setdefault("color", _colour_for(self.class_names[cls], i))
            plot_kwargs.setdefault("ls", "-")
            ax.plot(
                centres,
                frequency,
                label=_latex_label(self.class_names[cls]),
                **plot_kwargs,
            )

        ax.set_xlabel("radius (physical units)")
        ax.set_ylabel("fraction within bin" if per_bin else "fraction of all orbits")
        ax.legend(title="orbit class")
        return ax

    def plot_class_histograms(self, ax=None, **kwargs):
        """Bar chart of the number of orbits in each class.

        Draws one bar per populated class, with the class name as a categorical
        x-axis label and the bar height the number of orbits (from
        :meth:`counts`).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. A new figure and axes are created if omitted.
        **kwargs
            Passed through to ``ax.bar``. Pass ``color=...`` to override
            :data:`DEFAULT_PALETTE`.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes the bars were drawn on.
        """
        counts = self.counts()
        names = list(counts.keys())
        values = [counts[name] for name in names]

        if ax is None:
            _, ax = plt.subplots()

        positions = np.arange(len(names))
        bar_kwargs = dict(kwargs)
        bar_kwargs.setdefault(
            "color", [_colour_for(name, i) for i, name in enumerate(names)]
        )
        ax.bar(positions, values, **bar_kwargs)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [_latex_label(name) for name in names], rotation=45, ha="right"
        )
        ax.set_xlabel("orbit class")
        ax.set_ylabel("number of orbits")
        # Reserve room for the rotated tick labels so they are not clipped when
        # the figure is saved with a plain savefig() (no bbox_inches="tight").
        ax.figure.tight_layout()
        return ax

    def plot_frequency_map(
        self,
        ax=None,
        colourmap: Optional[str] = None,
        marker_size: float = 6.0,
        alpha: float = 0.8,
        legend: bool = True,
    ):
        """Frequency map: scatter of the fundamental-frequency ratios by class.

        Each orbit is plotted at ``(|w_x| / |w_z|, |w_y| / |w_z|)`` and coloured
        by its orbit class. In such a map regular families cluster and low-order
        resonances trace straight lines, so it is a compact visual summary of the
        orbital structure. Requires frequency data (the classification must come
        from :func:`analyse_family` / :func:`analyse_states`).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. A new figure and axes are created if omitted.
        colourmap : str, optional
            Name of a matplotlib colormap; each class takes the colour at its
            integer label, so colours are stable across plots. Defaults to
            :data:`DEFAULT_PALETTE` when omitted.
        marker_size : float, optional
            Scatter marker size (points**2).
        alpha : float, optional
            Marker opacity (helps with dense, overlapping points).
        legend : bool, optional
            Whether to draw the class legend.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes the points were drawn on.

        Raises
        ------
        ValueError
            If the classification carries no frequency data.
        """
        if self.fundamentals is None:
            raise ValueError(
                "no frequency data; classify results from analyse_family/"
                "analyse_states to populate the fundamental frequencies."
            )
        w = np.abs(np.asarray(self.fundamentals, dtype=np.float64))
        wz = w[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_x = w[:, 0] / wz
            ratio_y = w[:, 1] / wz
        finite = (wz > 0) & np.isfinite(ratio_x) & np.isfinite(ratio_y)

        if ax is None:
            _, ax = plt.subplots()
        cmap = plt.get_cmap(colourmap) if colourmap is not None else None
        for i, cls in enumerate(sorted(int(v) for v in np.unique(self.labels))):
            sel = finite & (self.labels == cls)
            if not np.any(sel):
                continue
            colour = (
                cmap(cls % cmap.N)
                if cmap is not None
                else _colour_for(self.class_names[cls], i)
            )
            ax.scatter(
                ratio_x[sel],
                ratio_y[sel],
                s=marker_size,
                alpha=alpha,
                color=colour,
                edgecolors="none",
                label=_latex_label(self.class_names[cls]),
            )
        ax.set_xlabel(r"$|\omega_x| / |\omega_z|$")
        ax.set_ylabel(r"$|\omega_y| / |\omega_z|$")
        if legend:
            ax.legend(
                title="orbit class", markerscale=2.0, framealpha=0.9, loc="upper left"
            )
        ax.figure.tight_layout()
        return ax

    def compare(
        self, other: Union["OrbitClassification", Iterable["OrbitClassification"]]
    ) -> "ClassificationComparison":
        """Compare per-particle classifications across two or more snapshots.

        Particles are matched by their ID (:attr:`ids`), so only those present
        in *every* classification are compared; particles missing from any one
        snapshot are dropped. This is useful for tracking how orbit families
        change as a system evolves, either between two snapshots or across a
        whole sequence of them (e.g. to draw a multi-column
        :meth:`ClassificationComparison.plot_sankey`).

        No temporal order is assumed: ``self`` is treated as the first stage
        and ``other`` (or each element of ``other``, in order) as the
        subsequent stages, purely by convention -- it is left to the caller to
        pass them in the order they should be compared.

        Parameters
        ----------
        other : OrbitClassification or iterable of OrbitClassification
            The classification(s) to compare against. A single
            :class:`OrbitClassification` compares two stages (as before); an
            iterable of them compares ``self`` followed by each item, in
            order, producing one stage per classification.

        Returns
        -------
        comparison : ClassificationComparison
            Per-particle labels/names at each stage for the matched
            particles, a mask of which particles changed family at any point
            in the sequence, and the family transitions between consecutive
            stages.

        Raises
        ------
        ValueError
            If ``other`` is an empty iterable, if any classification carries
            no particle IDs, or if the classifications use different class
            schemes (e.g. comparing a full classification against a
            condensed one).
        """
        if isinstance(other, OrbitClassification):
            others = [other]
        else:
            others = list(other)
            if not others:
                raise ValueError(
                    "other must be an OrbitClassification or a non-empty "
                    "iterable of them."
                )
        states = [self] + others

        for state in states:
            if state.ids is None:
                raise ValueError(
                    "all classifications must carry particle IDs to be "
                    "compared; build them with classify_orbits (which "
                    "records ids)."
                )
        for state in others:
            if state.class_names != self.class_names:
                raise ValueError(
                    "cannot compare classifications with different class "
                    "schemes (e.g. a full classification against a condensed "
                    "one); condense all of them with condense_families() "
                    "first, or compare classifications that all use the "
                    "same scheme."
                )

        ids_arrays = [np.asarray(state.ids) for state in states]
        common = ids_arrays[0]
        for ids in ids_arrays[1:]:
            common = np.intersect1d(common, ids)

        idx_list = []
        for ids in ids_arrays:
            order = np.argsort(ids)
            idx_list.append(order[np.searchsorted(ids[order], common)])

        sizes = "/".join(str(len(ids)) for ids in ids_arrays)
        logger.info(
            f"Comparing {len(states)} classifications: {len(common)} matched "
            f"of {sizes} particles"
        )

        labels = [np.asarray(state.labels)[idx] for state, idx in zip(states, idx_list)]
        names = [
            np.array([state.class_names[int(v)] for v in lab])
            for state, lab in zip(states, labels)
        ]
        changed = np.zeros(len(common), dtype=bool)
        for before, after in zip(names[:-1], names[1:]):
            changed |= before != after

        return ClassificationComparison(
            ids=common,
            labels=labels,
            names=names,
            changed=changed,
            class_names=[dict(state.class_names) for state in states],
        )


@dataclass
class ClassificationComparison:
    """Per-particle comparison of two or more :class:`OrbitClassification` stages.

    Produced by :meth:`OrbitClassification.compare`. All per-particle arrays
    are aligned and indexed identically, ordered by matched particle ID.
    ``labels``/``names``/``class_names`` hold one entry per stage, in the
    order passed to :meth:`OrbitClassification.compare` (``self`` first, then
    each ``other``). ``before``/``after`` refer to the first and last stage
    respectively, by convention only, and remain available as a convenience
    for the common two-stage comparison.

    Parameters
    ----------
    ids : numpy.ndarray
        (M,) particle IDs present in every stage, sorted ascending.
    labels : list of numpy.ndarray
        One (M,) array of integer class labels per stage.
    names : list of numpy.ndarray
        One (M,) array of class-name strings per stage, corresponding to
        ``labels``.
    changed : numpy.ndarray
        (M,) boolean, ``True`` where a particle's family name differs between
        any two consecutive stages.
    class_names : list of dict
        One label-to-name mapping per stage (used to build the transition
        matrices).
    """

    ids: np.ndarray
    labels: List[np.ndarray]
    names: List[np.ndarray]
    changed: np.ndarray
    class_names: List[Dict[int, str]]

    @property
    def n_matched(self) -> int:
        """Number of particles present in every stage.

        Returns
        -------
        n_matched : int
            Count of matched particles.
        """
        return int(len(self.ids))

    @property
    def n_stages(self) -> int:
        """Number of classification stages being compared.

        Returns
        -------
        n_stages : int
            ``2`` for a plain before/after comparison, more for a comparison
            built from an iterable of classifications.
        """
        return len(self.labels)

    @property
    def fraction_changed(self) -> float:
        """Fraction of matched particles that changed family at any point.

        Returns
        -------
        fraction_changed : float
            ``changed.sum() / n_matched`` (0.0 if no particles matched).
        """
        if self.n_matched == 0:
            return 0.0
        return float(np.sum(self.changed) / self.n_matched)

    @property
    def labels_before(self) -> np.ndarray:
        """Integer class labels at the first stage (see :attr:`labels`)."""
        return self.labels[0]

    @property
    def labels_after(self) -> np.ndarray:
        """Integer class labels at the last stage (see :attr:`labels`)."""
        return self.labels[-1]

    @property
    def names_before(self) -> np.ndarray:
        """Class names at the first stage (see :attr:`names`)."""
        return self.names[0]

    @property
    def names_after(self) -> np.ndarray:
        """Class names at the last stage (see :attr:`names`)."""
        return self.names[-1]

    @property
    def class_names_before(self) -> Dict[int, str]:
        """Label-to-name mapping at the first stage (see :attr:`class_names`)."""
        return self.class_names[0]

    @property
    def class_names_after(self) -> Dict[int, str]:
        """Label-to-name mapping at the last stage (see :attr:`class_names`)."""
        return self.class_names[-1]

    def transition_matrix(self, stage: int = 0):
        """Count matrix of family transitions between two consecutive stages.

        Parameters
        ----------
        stage : int, optional
            Index of the earlier stage in the pair to compare (``0`` is
            ``self`` vs the first ``other`` passed to
            :meth:`OrbitClassification.compare`). Defaults to ``0``, the only
            valid value for a plain two-stage (before/after) comparison.

        Returns
        -------
        row_names : list of str
            Class names of stage ``stage``, one per matrix row.
        col_names : list of str
            Class names of stage ``stage + 1``, one per matrix column.
        matrix : numpy.ndarray
            ``(len(row_names), len(col_names))`` integer counts, where
            ``matrix[i, j]`` is the number of particles classified as
            ``row_names[i]`` at stage ``stage`` and ``col_names[j]`` at stage
            ``stage + 1``.

        Raises
        ------
        ValueError
            If ``stage`` does not index a valid consecutive stage pair.
        """
        if not 0 <= stage < self.n_stages - 1:
            raise ValueError(
                f"stage must be in [0, {self.n_stages - 2}] for a comparison "
                f"with {self.n_stages} stages."
            )
        before_all = self.labels[stage]
        after_all = self.labels[stage + 1]
        before_names = self.class_names[stage]
        after_names = self.class_names[stage + 1]

        before_labels = sorted({int(v) for v in before_all})
        after_labels = sorted({int(v) for v in after_all})
        row_of = {lab: i for i, lab in enumerate(before_labels)}
        col_of = {lab: j for j, lab in enumerate(after_labels)}
        matrix = np.zeros((len(before_labels), len(after_labels)), dtype=np.int64)
        for before, after in zip(before_all, after_all):
            matrix[row_of[int(before)], col_of[int(after)]] += 1
        row_names = [before_names[lab] for lab in before_labels]
        col_names = [after_names[lab] for lab in after_labels]
        return row_names, col_names, matrix

    def plot_sankey(
        self,
        ax=None,
        colourmap: Optional[str] = None,
        node_width: float = 0.03,
        alpha: float = 0.6,
        stage_labels: Optional[list] = None,
    ):
        """Draw a Sankey diagram of the family flow across two or more stages.

        Draws one node column per classification stage (``self`` followed, in
        order, by each ``other`` passed to
        :meth:`OrbitClassification.compare`), with ribbons between every pair
        of consecutive columns whose width is proportional to the number of
        particles that moved from the source family to the destination
        family. Node heights are the family totals within that stage, so a
        family that keeps most of its members shows one dominant self-flow.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. A new figure and axes are created if omitted.
        colourmap : str, optional
            Name of a matplotlib colormap used to colour the families; each
            ribbon takes the colour of its source family. Defaults to
            :data:`DEFAULT_PALETTE` when omitted.
        node_width : float, optional
            Width of the node bars as a fraction of the horizontal span.
        alpha : float, optional
            Opacity of the flow ribbons.
        stage_labels : sequence of str, optional
            One label per stage, drawn above its node column. Defaults to
            ``["this", "other"]`` for a two-stage comparison, or
            ``["stage 0", "stage 1", ...]`` for more stages.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes the diagram was drawn on.

        Raises
        ------
        ValueError
            If ``stage_labels`` is given and does not have one entry per
            stage.
        """
        from matplotlib.patches import PathPatch, Rectangle
        from matplotlib.path import Path

        n_stages = self.n_stages
        if stage_labels is None:
            stage_labels = (
                ["this", "other"]
                if n_stages == 2
                else [f"stage {k}" for k in range(n_stages)]
            )
        elif len(stage_labels) != n_stages:
            raise ValueError(
                f"stage_labels must have {n_stages} entries (one per stage)."
            )

        transitions = [self.transition_matrix(k) for k in range(n_stages - 1)]
        total = float(transitions[0][2].sum())

        if ax is None:
            _, ax = plt.subplots()
        if total == 0:
            logger.warning("No matched particles; nothing to draw.")
            ax.axis("off")
            return ax
        gap = 0.02 * total  # vertical space between stacked nodes

        def _stack(sizes):
            # Top y of each node, stacked downward and centred about y = 0.
            column_height = sizes.sum() + gap * (len(sizes) - 1)
            tops = np.empty(len(sizes))
            y = 0.5 * column_height
            for i, h in enumerate(sizes):
                tops[i] = y
                y -= h + gap
            return tops

        # Names/sizes/tops for every stage. A stage's composition is read off
        # the row side of its outgoing transition (or the column side of its
        # incoming one for the last stage) -- both describe the same matched
        # population, since every transition is built from the same ids.
        stage_names = [transitions[0][0]] + [t[1] for t in transitions]
        stage_sizes = [transitions[0][2].sum(axis=1).astype(float)] + [
            t[2].sum(axis=0).astype(float) for t in transitions
        ]
        stage_tops = [_stack(sizes) for sizes in stage_sizes]

        # Consistent colour per family across all columns.
        names_all = list(dict.fromkeys(n for names in stage_names for n in names))
        if colourmap is not None:
            cmap = plt.get_cmap(colourmap)
            colours = {name: cmap(i % cmap.N) for i, name in enumerate(names_all)}
        else:
            colours = {name: _colour_for(name, i) for i, name in enumerate(names_all)}

        # Node column edges: the first/last columns sit flush with the plot
        # edges (as in the two-stage case); interior columns are centred on
        # their evenly spaced x position.
        x_positions = np.linspace(0.0, 1.0, n_stages)
        x_node_left = np.empty(n_stages)
        x_node_right = np.empty(n_stages)
        for k, x in enumerate(x_positions):
            if k == 0:
                x_node_left[k], x_node_right[k] = 0.0, node_width
            elif k == n_stages - 1:
                x_node_left[k], x_node_right[k] = 1.0 - node_width, 1.0
            else:
                x_node_left[k] = x - 0.5 * node_width
                x_node_right[k] = x + 0.5 * node_width

        # Ribbons: one pass per consecutive stage pair, outer loop over
        # source, inner over target, so each node's attachment points fill
        # top-down in a consistent order.
        for k, (row_names, col_names, matrix) in enumerate(transitions):
            x_left = x_node_right[k]  # right edge of the source column
            x_right = x_node_left[k + 1]  # left edge of the target column
            x_ctrl = 0.5 * (x_left + x_right)  # Bezier control x (smooth S-curve)
            left_cursor = stage_tops[k].copy()
            right_cursor = stage_tops[k + 1].copy()
            for i in range(len(row_names)):
                for j in range(len(col_names)):
                    flow = matrix[i, j]
                    if flow <= 0:
                        continue
                    yl_top, yl_bot = left_cursor[i], left_cursor[i] - flow
                    yr_top, yr_bot = right_cursor[j], right_cursor[j] - flow
                    vertices = [
                        (x_left, yl_top),
                        (x_ctrl, yl_top),
                        (x_ctrl, yr_top),
                        (x_right, yr_top),
                        (x_right, yr_bot),
                        (x_ctrl, yr_bot),
                        (x_ctrl, yl_bot),
                        (x_left, yl_bot),
                        (x_left, yl_top),
                    ]
                    codes = [
                        Path.MOVETO,
                        Path.CURVE4,
                        Path.CURVE4,
                        Path.CURVE4,
                        Path.LINETO,
                        Path.CURVE4,
                        Path.CURVE4,
                        Path.CURVE4,
                        Path.CLOSEPOLY,
                    ]
                    ax.add_patch(
                        PathPatch(
                            Path(vertices, codes),
                            facecolor=colours[row_names[i]],
                            edgecolor="none",
                            alpha=alpha,
                        )
                    )
                    left_cursor[i] = yl_bot
                    right_cursor[j] = yr_bot

        # Node bars. The first/last columns also get an external name+count
        # label (as in the two-stage case); interior columns rely on the
        # shared legend below, since ribbons on both sides leave no clear
        # spot for text.
        for k in range(n_stages):
            for name, size, top in zip(stage_names[k], stage_sizes[k], stage_tops[k]):
                ax.add_patch(
                    Rectangle(
                        (x_node_left[k], top - size),
                        x_node_right[k] - x_node_left[k],
                        size,
                        color=colours[name],
                    )
                )
                if k == 0:
                    ax.text(
                        x_node_left[k] - 0.01,
                        top - 0.5 * size,
                        f"{_latex_label(name)} ({int(size)})",
                        ha="right",
                        va="center",
                    )
                elif k == n_stages - 1:
                    ax.text(
                        x_node_right[k] + 0.01,
                        top - 0.5 * size,
                        f"{_latex_label(name)} ({int(size)})",
                        ha="left",
                        va="center",
                    )

        y_top = max(tops[0] for tops in stage_tops)
        y_bot = min(
            (tops[-1] - sizes[-1]).item()
            for tops, sizes in zip(stage_tops, stage_sizes)
        )
        for k in range(n_stages):
            x_centre = 0.5 * (x_node_left[k] + x_node_right[k])
            ax.text(x_centre, y_top + gap, stage_labels[k], ha="center", va="bottom")

        if n_stages > 2:
            from matplotlib.patches import Patch

            handles = [
                Patch(facecolor=colours[name], label=_latex_label(name))
                for name in names_all
            ]
            ax.legend(
                handles=handles,
                title="orbit class",
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=min(len(handles), 4),
            )

        margin = 0.05 * (y_top - y_bot)
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(y_bot - margin, y_top + gap + margin)
        ax.axis("off")
        return ax


def _primitive_resonance_vectors(max_order: int):
    """Primitive integer triples n, canonical sign, ordered by L1 order.

    The bound is on the L1 order ``|n_x| + |n_y| + |n_z|`` -- the physically
    meaningful resonance order -- not on the individual components, so only
    genuinely low-order commensurabilities (banana 2:-1:0, fish, pretzel, ...)
    qualify.

    Parameters
    ----------
    max_order : int
        Maximum L1 order ``|n|_1`` of the resonance vectors to generate.

    Returns
    -------
    vecs : numpy.ndarray
        (M, 3) float array of primitive resonance vectors, sorted by order.
    orders : numpy.ndarray
        (M,) L1 order of each vector, ascending.
    """
    out = []
    rng = range(-max_order, max_order + 1)
    for nx in rng:
        for ny in rng:
            for nz in rng:
                order = abs(nx) + abs(ny) + abs(nz)
                if order == 0 or order > max_order:
                    continue
                if gcd(gcd(abs(nx), abs(ny)), abs(nz)) != 1:
                    continue
                first = nx if nx != 0 else (ny if ny != 0 else nz)
                if first < 0:  # canonical sign
                    continue
                out.append((order, (nx, ny, nz)))
    out.sort(key=lambda t: t[0])
    orders = np.array([o for o, _ in out], dtype=np.int64)
    vecs = np.array([v for _, v in out], dtype=np.float64)  # (M,3)
    return vecs, orders


def _lattice_vectors(max_order: int) -> np.ndarray:
    """All non-zero integer triples with L1 order up to ``max_order``.

    Unlike :func:`_primitive_resonance_vectors`, this returns the full integer
    lattice (both signs, non-primitive included) used to test whether a spectral
    line is an integer *combination* of the base frequencies.

    Parameters
    ----------
    max_order : int
        Maximum L1 order ``|n_x| + |n_y| + |n_z|`` of the vectors.

    Returns
    -------
    vecs : numpy.ndarray
        (M, 3) float array of integer combination vectors.
    """
    rng = range(-max_order, max_order + 1)
    out = [
        (nx, ny, nz)
        for nx in rng
        for ny in rng
        for nz in rng
        if 1 <= abs(nx) + abs(ny) + abs(nz) <= max_order
    ]
    return np.array(out, dtype=np.float64)


def _lattice_vectors_2d(max_order: int) -> np.ndarray:
    """Non-zero integer pairs with L1 order up to ``max_order`` (see below).

    The two-base analogue of :func:`_lattice_vectors`, used when reducing a line
    to a combination of the first two base frequencies.

    Parameters
    ----------
    max_order : int
        Maximum L1 order ``|n_0| + |n_1|`` of the vectors.

    Returns
    -------
    vecs : numpy.ndarray
        (P, 2) float array of integer combination vectors.
    """
    rng = range(-max_order, max_order + 1)
    out = [(a, b) for a in rng for b in rng if 1 <= abs(a) + abs(b) <= max_order]
    return np.array(out, dtype=np.float64)


def _resolve_n_workers(n_workers: Optional[int]) -> int:
    """Resolve a worker count for the classification chunk thread pools.

    Defaults to serial (``1``): the dominant reduction (:func:`_detect_irregular`)
    is memory-bandwidth-bound over large broadcast temporaries, and profiling on
    real multi-million-orbit data showed a thread pool adds contention rather
    than throughput -- every thread count tried (4/8/16) was slower than serial,
    getting worse as threads increased. Pass ``n_workers`` explicitly to opt in
    and measure it on your own workload/hardware, where it may behave
    differently.
    """
    if n_workers is not None:
        return max(1, int(n_workers))
    return 1


def _map_chunks(n, chunk, worker, n_workers):
    """Apply ``worker(lo, hi)`` to disjoint ``[lo, hi)`` ranges covering ``[0, n)``.

    Runs serially if there is only one chunk or ``n_workers <= 1``. Otherwise
    the (independent, read-only-input) chunks are dispatched to a thread pool:
    numpy releases the GIL for the large elementwise/matmul work each chunk
    does here, so this scales across cores despite the GIL.

    Returns
    -------
    results : list of (lo, hi, value)
        One entry per chunk, in order, where ``value`` is ``worker(lo, hi)``.
    """
    ranges = [(lo, min(lo + chunk, n)) for lo in range(0, n, chunk)]
    if n_workers <= 1 or len(ranges) <= 1:
        return [(lo, hi, worker(lo, hi)) for lo, hi in ranges]
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [(lo, hi, pool.submit(worker, lo, hi)) for lo, hi in ranges]
        return [(lo, hi, fut.result()) for lo, hi, fut in futures]


def _detect_irregular(fundamentals, lines, amp_frac, tol, max_order, n_workers=None):
    """Flag orbits needing more than three base frequencies (irregular).

    Following Frigo et al. (2021) / Carpintero & Aguilar (1998), an orbit is
    *irregular* when its spectrum cannot be described by three base frequencies.
    The base frequencies are identified greedily from the reduced spectral lines:
    the strongest *significant* line (amplitude at least ``amp_frac`` of the
    orbit's strongest line) is the first base; the strongest line that is not an
    integer combination of the bases so far becomes the next base; and so on. If
    a fourth independent base frequency is found the orbit is irregular. A line
    is an integer combination when it lies within ``tol * max|w_fund|`` of
    ``n_0 b_0 + n_1 b_1 + n_2 b_2`` for integers with ``|n|_1 <= max_order``.

    Parameters
    ----------
    fundamentals : numpy.ndarray
        (N, 3) signed fundamental frequency per axis (used for the frequency
        scale).
    lines : numpy.ndarray
        (N, 3, n_lines, 2) leading (frequency, amplitude) spectral lines.
    amp_frac : float
        A line counts as significant if its amplitude is at least this fraction
        of the orbit's largest line amplitude.
    tol : float
        Relative tolerance for the integer-combination test.
    max_order : int
        Maximum L1 order of the integer combinations tested.
    n_workers : int, optional
        Number of threads to process chunks with; chunks are independent, but
        this reduction is memory-bandwidth-bound, so threading was measured to
        *hurt* on real data rather than help (see :func:`_resolve_n_workers`).
        Defaults to serial (``1``); pass this explicitly only if you've
        measured a benefit on your own workload.

    Returns
    -------
    irregular : numpy.ndarray
        (N,) boolean, True where the orbit is irregular (> 3 base frequencies).
    """
    fundamentals = np.asarray(fundamentals)
    lines = np.asarray(lines)
    # Work in the results' own float dtype (float32 once round-tripped through
    # save/load) rather than forcing float64: this reduction is the dominant
    # cost of classification and is elementwise/bandwidth-bound, so narrowing
    # the dtype roughly halves it, and the tolerances used here don't need
    # float64 precision.
    dtype = np.result_type(fundamentals.dtype, lines.dtype)
    f = np.abs(fundamentals).astype(dtype, copy=False)  # (N,3)
    freqs = lines[..., 0].astype(dtype, copy=False)  # (N,3,nl)
    amps = lines[..., 1].astype(dtype, copy=False)  # (N,3,nl)
    n = f.shape[0]
    cols = freqs.shape[1] * freqs.shape[2]
    line_abs = np.abs(freqs.reshape(n, cols))  # (N,C) magnitudes
    line_amp = amps.reshape(n, cols)  # (N,C)

    scale = np.maximum(np.max(f, axis=1), 1e-30)  # (N,) frequency scale
    amp_max = np.maximum(np.max(line_amp, axis=1), 1e-30)  # (N,)

    mult1 = np.arange(1, max_order + 1, dtype=dtype)  # (K,) single-base
    vecs2 = _lattice_vectors_2d(max_order).astype(dtype, copy=False)  # (P,2)
    vecs3 = _lattice_vectors(max_order).astype(dtype, copy=False)  # (Q,3)

    def _reducible(la_abs, combos, tol_abs):
        # Nearest integer-combination distance per line <= tolerance.
        resid = np.abs(la_abs[:, :, None] - combos[:, None, :])  # (m, C, n_combos)
        return resid.min(axis=2) <= tol_abs  # (m, C)

    def _process(lo, hi):
        la = line_abs[lo:hi]  # (m, C)
        amp = line_amp[lo:hi]  # (m, C)
        tol_abs = tol * scale[lo:hi, None]  # (m, 1)
        rows = np.arange(hi - lo)
        significant = (amp >= amp_frac * amp_max[lo:hi, None]) & (la > tol_abs)

        def _pick(candidate):
            # Frequency of the strongest candidate line per orbit (0 if none).
            present = candidate.any(axis=1)
            idx = np.argmax(np.where(candidate, amp, -1.0), axis=1)
            return np.where(present, la[rows, idx], 0.0), present

        # First base: the strongest significant line.
        b0, has0 = _pick(significant)

        # Second base: strongest line not a multiple of b0.
        red0 = _reducible(la, b0[:, None] * mult1[None, :], tol_abs)
        b1, has1 = _pick(significant & ~red0)

        # Third base: strongest line not a combination of {b0, b1}.
        combos2 = np.stack([b0, b1], axis=1) @ vecs2.T  # (m, P)
        red1 = _reducible(la, combos2, tol_abs)
        b2, has2 = _pick(significant & ~red1)

        # Irregular: a fourth independent line beyond {b0, b1, b2}.
        combos3 = np.stack([b0, b1, b2], axis=1) @ vecs3.T  # (m, Q)
        red2 = _reducible(la, combos3, tol_abs)
        return has2 & np.any(significant & ~red2, axis=1)

    chunk = max(256, int(5_000_000 // max(1, cols * len(vecs3))))
    irregular = np.zeros(n, dtype=bool)
    for lo, hi, result in _map_chunks(
        n, chunk, _process, _resolve_n_workers(n_workers)
    ):
        irregular[lo:hi] = result
    return irregular


def _find_resonances(w, max_order, tol, chunk=20000, n_workers=None):
    """Find the lowest-order commensurability of each frequency triple.

    A commensurability is a low-order integer vector ``n`` with
    ``|n . w| / max|w| < tol``.

    Parameters
    ----------
    w : numpy.ndarray
        (N, 3) absolute fundamental frequencies per orbit.
    max_order : int
        Maximum L1 order of the resonance vectors to consider.
    tol : float
        Relative tolerance ``|n . w| / max|w|`` for accepting a resonance.
    chunk : int, optional
        Number of orbits processed per block (bounds memory).
    n_workers : int, optional
        Number of threads to process chunks with; see :func:`_detect_irregular`.

    Returns
    -------
    vectors : numpy.ndarray
        (N, 3) integer resonance vector per orbit (all zero if none found).
    orders : numpy.ndarray
        (N,) L1 order of each resonance (0 if none found).
    """
    w = np.asarray(w)
    vecs, orders = _primitive_resonance_vectors(max_order)
    vecs = vecs.astype(w.dtype, copy=False)  # match w's dtype (see _detect_irregular)
    N = len(w)
    scale = np.maximum(np.max(np.abs(w), axis=1), 1e-30)  # (N,)

    def _process(lo, hi):
        # |n . w| / max|w| for every candidate; vecs are sorted by order, so the
        # first hit is the lowest-order commensurability.
        resid = np.abs(w[lo:hi] @ vecs.T)  # (m, M)
        norm = resid / scale[lo:hi, None]
        hit = norm < tol
        any_hit = hit.any(axis=1)
        first = np.argmax(hit, axis=1)  # first (lowest order)
        rv = np.zeros((hi - lo, 3), dtype=np.int64)
        ro = np.zeros(hi - lo, dtype=np.int64)
        rv[any_hit] = vecs[first[any_hit]].astype(np.int64)
        ro[any_hit] = orders[first[any_hit]]
        return rv, ro

    res_vec = np.zeros((N, 3), dtype=np.int64)
    res_ord = np.zeros(N, dtype=np.int64)
    for lo, hi, (rv, ro) in _map_chunks(
        N, chunk, _process, _resolve_n_workers(n_workers)
    ):
        res_vec[lo:hi] = rv
        res_ord[lo:hi] = ro
    return res_vec, res_ord


def classify_orbits(
    results,
    circ_thresh: float = 0.7,
    freq_tol: float = 0.01,
    amp_frac: float = 0.05,
    planar_thresh: float = 0.02,
    resonance_max_order: int = 5,
    resonance_tol: float = 0.01,
    inner_outer_ratio: float = 1.0,
    irregular_amp_frac: float = 0.1,
    irregular_tol: float = 0.02,
    irregular_max_order: int = 6,
    n_workers: Optional[int] = None,
) -> OrbitClassification:
    """Classify the orbits in an :class:`~lanfear.OrbitResults`.

    Parameters
    ----------
    results : lanfear.OrbitResults
        Orbits to classify. Frequency fields (``fundamentals``/``lines``) enable
        the 1:1:1 rosette and boxlet-resonance tests; without them the rosette
        test falls back to the shape-tensor planarity.
    circ_thresh : float, optional
        ``circ_a`` above this counts as circulation about axis ``a`` (a tube).
    freq_tol : float, optional
        A loop whose (amplitude-active) axis fundamentals are mutually equal to
        within this fraction is a rosette (1:1:1); a tube is 1:1:pi (two axes
        share a frequency, the circulation axis differs).
    amp_frac : float, optional
        An axis whose leading spectral amplitude exceeds this fraction of the
        largest is "active" (used to ignore silent axes in the 1:1:1 test).
    planar_thresh : float, optional
        Without frequency data, a loop with shape-tensor
        ``lambda_min / lambda_max`` below this is taken to be a (planar) rosette.
    resonance_max_order : int, optional
        Maximum L1 order searched for the boxlet commensurability.
    resonance_tol : float, optional
        Tolerance ``|n.w| / max|w|`` for accepting a boxlet resonance.
    inner_outer_ratio : float, optional
        Long-axis (x) tubes are split by morphology (Frigo et al. 2021): using the
        peak-``|y|`` ratio between an ``|x|`` centre strip and a border strip at
        the z=0 crossings (``x_tube_ratio``), a tube with a pinched waist
        (``x_tube_ratio`` below this) is *inner*, else *outer*.
    irregular_amp_frac : float, optional
        Amplitude threshold (as a fraction of an orbit's strongest line) above
        which a spectral line is tested for the irregular criterion. Requires
        frequency data.
    irregular_tol : float, optional
        Relative tolerance ``|w - n.w_fund| / max|w_fund|`` for deciding whether
        a line is an integer combination of the three fundamentals.
    irregular_max_order : int, optional
        Maximum L1 order of the integer combinations tested for the irregular
        criterion.
    n_workers : int, optional
        Threads used to classify chunks of orbits for the resonance and
        irregular tests. Defaults to serial (``1``); threading was measured to
        make this *slower* on real data (memory-bandwidth-bound), so only pass
        this if you've confirmed a benefit on your own workload/hardware.

    Returns
    -------
    classification : OrbitClassification
        Per-orbit family labels and the diagnostic quantities used to derive
        them (circulation, tube axis, planarity, resonance vector and order).

    Notes
    -----
    When frequency data is present, an orbit whose spectrum needs more than
    three base frequencies is labelled :attr:`OrbitClass.IRREGULAR` (Frigo et
    al. 2021), overriding the regular-family assignment: such an orbit is not
    confined to a regular 3-torus and is a likely-chaotic candidate.
    """
    c = results.column
    status = c("status")
    N = len(status)

    Lm = np.stack([c("Lx_mean"), c("Ly_mean"), c("Lz_mean")], axis=1)
    La = np.stack([c("Lx_abs_mean"), c("Ly_abs_mean"), c("Lz_abs_mean")], axis=1)
    circ = np.abs(Lm) / np.maximum(La, 1e-30)  # (N,3)
    tube_axis = np.argmax(circ, axis=1)
    n_circ = np.sum(circ > circ_thresh, axis=1)

    # Shape tensor -> planarity (smallest / largest eigenvalue).
    S = np.zeros((N, 3, 3))
    S[:, 0, 0] = c("Sxx")
    S[:, 1, 1] = c("Syy")
    S[:, 2, 2] = c("Szz")
    S[:, 0, 1] = S[:, 1, 0] = c("Sxy")
    S[:, 0, 2] = S[:, 2, 0] = c("Sxz")
    S[:, 1, 2] = S[:, 2, 1] = c("Syz")
    evals = np.linalg.eigvalsh(S)  # ascending (N,3)
    planarity = evals[:, 0] / np.maximum(evals[:, 2], 1e-30)

    # Rosette test: are the active-axis fundamentals mutually 1:1:1?
    res_vec = np.zeros((N, 3), dtype=np.int64)
    res_ord = np.zeros(N, dtype=np.int64)
    if results.fundamentals is not None:
        w = np.abs(results.fundamentals)  # (N,3)
        amp = (
            results.lines[:, :, 0, 1] if results.lines is not None else np.ones_like(w)
        )
        active = amp > amp_frac * np.max(amp, axis=1, keepdims=True)
        n_active = np.sum(active, axis=1)
        w_hi = np.where(active, w, -np.inf).max(axis=1)
        w_lo = np.where(active, w, np.inf).min(axis=1)
        freq_111 = (n_active >= 2) & (
            w_hi <= (1.0 + freq_tol) * np.maximum(w_lo, 1e-30)
        )
        res_vec, res_ord = _find_resonances(
            w, resonance_max_order, resonance_tol, n_workers=n_workers
        )
    else:
        # No frequency data: fall back to the shape-tensor planarity.
        logger.debug(
            "No frequency data; using shape-tensor planarity for the rosette test"
        )
        freq_111 = planarity < planar_thresh

    labels = np.full(N, OrbitClass.UNCLASSIFIED, dtype=np.int64)
    ok = status == 0
    is_loop = n_circ >= 1

    # Rosette: a loop whose axis fundamentals are all commensurate 1:1:1.
    labels[ok & is_loop & freq_111] = OrbitClass.ROSETTE

    # Tubes: a single circulation axis, not 1:1:1.
    tube = ok & is_loop & ~freq_111
    labels[tube & (tube_axis == 2)] = OrbitClass.SHORT_AXIS_TUBE
    labels[tube & (tube_axis == 1)] = OrbitClass.INTERMEDIATE_AXIS_TUBE
    # Long-axis (x) tubes split inner/outer by morphology (Frigo et al. 2021,
    # after orbit-analysis): an inner x-tube is pinched at the waist -- at its
    # z=0 crossings the y-extent peaks at the x-ends, not the centre, so the
    # centre/border peak-|y| ratio is < 1. An outer x-tube is widest at the
    # centre (ratio >= 1).
    lat = tube & (tube_axis == 0)
    x_tube_ratio = c("x_tube_ratio")
    labels[lat & (x_tube_ratio < inner_outer_ratio)] = OrbitClass.INNER_LONG_AXIS_TUBE
    labels[lat & (x_tube_ratio >= inner_outer_ratio)] = OrbitClass.OUTER_LONG_AXIS_TUBE

    # Boxes: no circulation. A low-order (>=2) resonance marks a boxlet.
    box = ok & ~is_loop
    labels[box] = OrbitClass.PIBOX
    labels[box & (res_ord >= 2)] = OrbitClass.BOXLET

    # Irregular (Frigo et al. 2021): a spectrum needing > 3 base frequencies.
    # Determined from the spectral lines, it overrides the regular-family label.
    if results.fundamentals is not None and results.lines is not None:
        irregular = _detect_irregular(
            results.fundamentals,
            results.lines,
            irregular_amp_frac,
            irregular_tol,
            irregular_max_order,
            n_workers=n_workers,
        )
        labels[ok & irregular] = OrbitClass.IRREGULAR

    counts = {
        CLASS_NAMES[int(v)]: int(cnt)
        for v, cnt in zip(*np.unique(labels, return_counts=True))
    }
    logger.info(f"Classified {N} orbits: {counts}")

    # Radii are reported in physical units. Both the snapshot radius and the
    # orbit-averaged radius r_mean are stored in HO units, so scale them by the
    # length unit (the scale radius). The snapshot radius is the default binning
    # radius for plot_class_fractions.
    length_unit = results.length_unit
    radius = np.asarray(results.initial_radius, dtype=float) * length_unit
    radius_orbit_averaged = c("r_mean") * length_unit

    return OrbitClassification(
        labels=labels,
        circulation=circ,
        tube_axis=tube_axis,
        planarity=planarity,
        resonance=res_vec,
        resonance_order=res_ord,
        radius=radius,
        radius_orbit_averaged=radius_orbit_averaged,
        ids=results.ids,
        fundamentals=results.fundamentals,
    )
