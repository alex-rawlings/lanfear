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

from dataclasses import dataclass, field
from enum import IntEnum
from math import gcd
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

from ._logging import get_logger

logger = get_logger(__name__)


class OrbitClass(IntEnum):
    """Enumeration of orbit families assigned by :func:`classify_orbits`."""

    UNCLASSIFIED = 0
    PIBOX = 1
    BOXLET = 2  # resonant box (banana/fish/pretzel/...)
    SHORT_AXIS_TUBE = 3  # z-tube
    INNER_LONG_AXIS_TUBE = 4  # inner x-tube
    OUTER_LONG_AXIS_TUBE = 5  # outer x-tube
    INTERMEDIATE_AXIS_TUBE = 6  # y-tube (generally unstable)
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
    "short_axis_tube": r"$z\mathrm{-tube}$",
    "inner_long_axis_tube": r"$\mathrm{inner}\;x\mathrm{-tube}$",
    "outer_long_axis_tube": r"$\mathrm{outer}\;x\mathrm{-tube}$",
    "intermediate_axis_tube": r"$y\mathrm{-tube}$",
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
        self,
        edges,
        per_bin: bool = True,
        radius=None,
        ax=None,
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

        for cls in sorted(int(v) for v in np.unique(self.labels)):
            selected = in_range & (self.labels == cls)
            count = np.bincount(bin_index[selected], minlength=n_bins).astype(float)
            if per_bin:
                normalisation = bin_total.copy()
                # Empty bins have no defined composition -> leave a gap (NaN).
                normalisation[normalisation == 0] = np.nan
                frequency = count / normalisation
            else:
                frequency = count / grand_total if grand_total > 0 else count
            ax.plot(
                centres,
                frequency,
                marker="o",
                label=_latex_label(self.class_names[cls]),
            )

        ax.set_xlabel("radius (physical units)")
        ax.set_ylabel("fraction within bin" if per_bin else "fraction of all orbits")
        ax.legend(title="orbit class")
        return ax

    def plot_class_histograms(self, ax=None):
        """Bar chart of the number of orbits in each class.

        Draws one bar per populated class, with the class name as a categorical
        x-axis label and the bar height the number of orbits (from
        :meth:`counts`).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. A new figure and axes are created if omitted.

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
        ax.bar(positions, values)
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
        colourmap: str = "tab10",
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
            Name of the matplotlib colormap; each class takes the colour at its
            integer label, so colours are stable across plots.
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
        cmap = plt.get_cmap(colourmap)
        for cls in sorted(int(v) for v in np.unique(self.labels)):
            sel = finite & (self.labels == cls)
            if not np.any(sel):
                continue
            ax.scatter(
                ratio_x[sel],
                ratio_y[sel],
                s=marker_size,
                alpha=alpha,
                color=cmap(cls % cmap.N),
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

    def compare(self, other: "OrbitClassification") -> "ClassificationComparison":
        """Compare per-particle classifications between two snapshots.

        Particles are matched by their ID (:attr:`ids`), so only those present
        in *both* classifications are compared; particles present in only one
        are dropped. This is useful for tracking how orbit families change when
        a system is perturbed and allowed to settle.

        No temporal order is assumed: ``self`` is treated as the *before* state
        and ``other`` as the *after* state purely by convention, and it is left
        to the caller to decide which snapshot is the earlier one.

        Parameters
        ----------
        other : OrbitClassification
            The classification to compare against (the *after* state).

        Returns
        -------
        comparison : ClassificationComparison
            Per-particle before/after labels for the matched particles, a mask
            of which particles changed family, and the family transitions.

        Raises
        ------
        ValueError
            If either classification carries no particle IDs, or if the two
            classifications use different class schemes (e.g. comparing a full
            classification against a condensed one).
        """
        if self.ids is None or other.ids is None:
            raise ValueError(
                "both classifications must carry particle IDs to be compared; "
                "build them with classify_orbits (which records ids)."
            )
        if self.class_names != other.class_names:
            raise ValueError(
                "cannot compare classifications with different class schemes "
                "(e.g. a full classification against a condensed one); condense "
                "both with condense_families() first, or compare two full "
                "classifications."
            )
        ids_self = np.asarray(self.ids)
        ids_other = np.asarray(other.ids)
        common, idx_self, idx_other = np.intersect1d(
            ids_self, ids_other, return_indices=True
        )
        n_self, n_other = len(ids_self), len(ids_other)
        logger.info(
            "Comparing classifications: %d matched of %d / %d particles "
            "(%d only-before, %d only-after dropped)",
            len(common),
            n_self,
            n_other,
            n_self - len(common),
            n_other - len(common),
        )

        labels_before = np.asarray(self.labels)[idx_self]
        labels_after = np.asarray(other.labels)[idx_other]
        names_before = np.array([self.class_names[int(v)] for v in labels_before])
        names_after = np.array([other.class_names[int(v)] for v in labels_after])

        return ClassificationComparison(
            ids=common,
            labels_before=labels_before,
            labels_after=labels_after,
            names_before=names_before,
            names_after=names_after,
            changed=names_before != names_after,
            class_names_before=dict(self.class_names),
            class_names_after=dict(other.class_names),
        )


@dataclass
class ClassificationComparison:
    """Per-particle comparison of two :class:`OrbitClassification` snapshots.

    Produced by :meth:`OrbitClassification.compare`. All per-particle arrays are
    aligned and indexed identically, ordered by matched particle ID. ``before``
    refers to the classification the method was called on and ``after`` to its
    argument, by convention only.

    Parameters
    ----------
    ids : numpy.ndarray
        (M,) particle IDs present in both classifications, sorted ascending.
    labels_before, labels_after : numpy.ndarray
        (M,) integer class labels of each matched particle in the two snapshots.
    names_before, names_after : numpy.ndarray
        (M,) class-name strings corresponding to ``labels_before``/``after``.
    changed : numpy.ndarray
        (M,) boolean, ``True`` where a particle's family name differs between
        the snapshots.
    class_names_before, class_names_after : dict
        Label-to-name mappings of the two classifications (used to build the
        transition matrix).
    """

    ids: np.ndarray
    labels_before: np.ndarray
    labels_after: np.ndarray
    names_before: np.ndarray
    names_after: np.ndarray
    changed: np.ndarray
    class_names_before: Dict[int, str]
    class_names_after: Dict[int, str]

    @property
    def n_matched(self) -> int:
        """Number of particles present in both classifications.

        Returns
        -------
        n_matched : int
            Count of matched particles.
        """
        return int(len(self.ids))

    @property
    def fraction_changed(self) -> float:
        """Fraction of matched particles that changed family.

        Returns
        -------
        fraction_changed : float
            ``changed.sum() / n_matched`` (0.0 if no particles matched).
        """
        if self.n_matched == 0:
            return 0.0
        return float(np.sum(self.changed) / self.n_matched)

    def transition_matrix(self):
        """Count matrix of family transitions from before to after.

        Returns
        -------
        row_names : list of str
            Class names of the *before* state, one per matrix row.
        col_names : list of str
            Class names of the *after* state, one per matrix column.
        matrix : numpy.ndarray
            ``(len(row_names), len(col_names))`` integer counts, where
            ``matrix[i, j]`` is the number of particles classified as
            ``row_names[i]`` before and ``col_names[j]`` after.
        """
        before_labels = sorted({int(v) for v in self.labels_before})
        after_labels = sorted({int(v) for v in self.labels_after})
        row_of = {lab: i for i, lab in enumerate(before_labels)}
        col_of = {lab: j for j, lab in enumerate(after_labels)}
        matrix = np.zeros((len(before_labels), len(after_labels)), dtype=np.int64)
        for before, after in zip(self.labels_before, self.labels_after):
            matrix[row_of[int(before)], col_of[int(after)]] += 1
        row_names = [self.class_names_before[lab] for lab in before_labels]
        col_names = [self.class_names_after[lab] for lab in after_labels]
        return row_names, col_names, matrix

    def plot_sankey(
        self,
        ax=None,
        colourmap: str = "tab20",
        node_width: float = 0.03,
        alpha: float = 0.6,
    ):
        """Draw a Sankey diagram of the family flow from *before* to *after*.

        Left-hand nodes are the *before* families (``this`` classification) and
        right-hand nodes the *after* families (``other``); the ribbon joining a
        left node to a right node has a width proportional to the number of
        particles that moved from the first family to the second. Node heights
        are the family totals, so a family that keeps most of its members shows
        one dominant self-flow.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. A new figure and axes are created if omitted.
        colourmap : str, optional
            Name of the matplotlib colormap used to colour the families; each
            ribbon takes the colour of its source (before) family.
        node_width : float, optional
            Width of the node bars as a fraction of the horizontal span.
        alpha : float, optional
            Opacity of the flow ribbons.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes the diagram was drawn on.
        """
        from matplotlib.patches import PathPatch, Rectangle
        from matplotlib.path import Path

        row_names, col_names, matrix = self.transition_matrix()
        total = float(matrix.sum())

        if ax is None:
            _, ax = plt.subplots()
        if total == 0:
            logger.warning("No matched particles; nothing to draw.")
            ax.axis("off")
            return ax

        # Node heights are the family totals (row/column sums of the flow).
        left_sizes = matrix.sum(axis=1).astype(float)
        right_sizes = matrix.sum(axis=0).astype(float)
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

        left_top = _stack(left_sizes)
        right_top = _stack(right_sizes)

        # Consistent colour per family across both columns.
        names_all = list(dict.fromkeys(list(row_names) + list(col_names)))
        cmap = plt.get_cmap(colourmap)
        colours = {name: cmap(i % cmap.N) for i, name in enumerate(names_all)}

        x_left = node_width  # right edge of the left column (ribbon start)
        x_right = 1.0 - node_width  # left edge of the right column (ribbon end)
        x_ctrl = 0.5 * (x_left + x_right)  # Bezier control x (smooth S-curve)

        # Ribbons: outer loop over source, inner over target, so each node's
        # attachment points fill top-down in a consistent order.
        left_cursor = left_top.copy()
        right_cursor = right_top.copy()
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

        # Node bars and labels.
        for name, size, top in zip(row_names, left_sizes, left_top):
            ax.add_patch(
                Rectangle((0.0, top - size), node_width, size, color=colours[name])
            )
            ax.text(
                -0.01,
                top - 0.5 * size,
                f"{_latex_label(name)} ({int(size)})",
                ha="right",
                va="center",
            )
        for name, size, top in zip(col_names, right_sizes, right_top):
            ax.add_patch(
                Rectangle(
                    (1.0 - node_width, top - size),
                    node_width,
                    size,
                    color=colours[name],
                )
            )
            ax.text(
                1.0 + 0.01,
                top - 0.5 * size,
                f"{_latex_label(name)} ({int(size)})",
                ha="left",
                va="center",
            )

        y_top = max(left_top[0], right_top[0])
        y_bot = min(left_top[-1] - left_sizes[-1], right_top[-1] - right_sizes[-1])
        ax.text(0.5 * node_width, y_top + gap, "this", ha="center", va="bottom")
        ax.text(1.0 - 0.5 * node_width, y_top + gap, "other", ha="center", va="bottom")

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


def _detect_irregular(fundamentals, lines, amp_frac, tol, max_order):
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

    Returns
    -------
    irregular : numpy.ndarray
        (N,) boolean, True where the orbit is irregular (> 3 base frequencies).
    """
    f = np.abs(np.asarray(fundamentals, dtype=np.float64))  # (N,3)
    freqs = np.asarray(lines, dtype=np.float64)[..., 0]  # (N,3,nl)
    amps = np.asarray(lines, dtype=np.float64)[..., 1]  # (N,3,nl)
    n = f.shape[0]
    cols = freqs.shape[1] * freqs.shape[2]
    line_abs = np.abs(freqs.reshape(n, cols))  # (N,C) magnitudes
    line_amp = amps.reshape(n, cols)  # (N,C)

    scale = np.maximum(np.max(f, axis=1), 1e-30)  # (N,) frequency scale
    amp_max = np.maximum(np.max(line_amp, axis=1), 1e-30)  # (N,)

    mult1 = np.arange(1, max_order + 1, dtype=np.float64)  # (K,) single-base
    vecs2 = _lattice_vectors_2d(max_order)  # (P,2)
    vecs3 = _lattice_vectors(max_order)  # (Q,3)

    irregular = np.zeros(n, dtype=bool)
    chunk = max(256, int(5_000_000 // max(1, cols * len(vecs3))))

    def _reducible(la_abs, combos, tol_abs):
        # Nearest integer-combination distance per line <= tolerance.
        resid = np.abs(la_abs[:, :, None] - combos[:, None, :])  # (m, C, n_combos)
        return resid.min(axis=2) <= tol_abs  # (m, C)

    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
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
        irregular[lo:hi] = has2 & np.any(significant & ~red2, axis=1)

    return irregular


def _find_resonances(w, max_order, tol, chunk=20000):
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

    Returns
    -------
    vectors : numpy.ndarray
        (N, 3) integer resonance vector per orbit (all zero if none found).
    orders : numpy.ndarray
        (N,) L1 order of each resonance (0 if none found).
    """
    vecs, orders = _primitive_resonance_vectors(max_order)
    N = len(w)
    res_vec = np.zeros((N, 3), dtype=np.int64)
    res_ord = np.zeros(N, dtype=np.int64)
    scale = np.maximum(np.max(np.abs(w), axis=1), 1e-30)  # (N,)
    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        # |n . w| / max|w| for every candidate; vecs are sorted by order, so the
        # first hit is the lowest-order commensurability.
        resid = np.abs(w[lo:hi] @ vecs.T)  # (n, M)
        norm = resid / scale[lo:hi, None]
        hit = norm < tol
        any_hit = hit.any(axis=1)
        first = np.argmax(hit, axis=1)  # first (lowest order)
        idx = np.where(any_hit)[0]
        res_vec[lo + idx] = vecs[first[idx]].astype(np.int64)
        res_ord[lo + idx] = orders[first[idx]]
    return res_vec, res_ord


def classify_orbits(
    results,
    circ_thresh: float = 0.7,
    freq_tol: float = 0.05,
    amp_frac: float = 0.05,
    planar_thresh: float = 0.02,
    resonance_max_order: int = 5,
    resonance_tol: float = 0.01,
    inner_outer_ratio: float = 1.0,
    irregular_amp_frac: float = 0.1,
    irregular_tol: float = 0.02,
    irregular_max_order: int = 6,
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
        Long-axis (x) tubes are split by morphology (Frigo et al. 2021): using
        the peak-|y| ratio between an |x| centre strip and a border strip at the
        z=0 crossings (``x_tube_ratio``), a tube with a pinched waist
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
        res_vec, res_ord = _find_resonances(w, resonance_max_order, resonance_tol)
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
        )
        labels[ok & irregular] = OrbitClass.IRREGULAR

    counts = {
        CLASS_NAMES[int(v)]: int(cnt)
        for v, cnt in zip(*np.unique(labels, return_counts=True))
    }
    logger.info("Classified %d orbits: %s", N, counts)

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
