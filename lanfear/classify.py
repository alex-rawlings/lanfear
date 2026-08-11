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
    BOX = 1
    BOXLET = 2  # resonant box (banana/fish/pretzel/...)
    SHORT_AXIS_TUBE = 3  # z-tube
    INNER_LONG_AXIS_TUBE = 4  # inner x-tube
    OUTER_LONG_AXIS_TUBE = 5  # outer x-tube
    INTERMEDIATE_AXIS_TUBE = 6  # y-tube (generally unstable)
    ROSETTE = 7  # planar loop


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
_BOX_CLASSES = (OrbitClass.BOX, OrbitClass.BOXLET)


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
        (N,) characteristic radius of each orbit (the time-averaged radius
        ``r_mean``), recorded by :func:`classify_orbits` and used by
        :meth:`plot_class_fractions`. ``None`` when not available.
    ids : numpy.ndarray, optional
        (N,) particle ID of each orbit, recorded by :func:`classify_orbits` and
        used by :meth:`compare` to match particles between two classifications.
        ``None`` when not available.
    """

    labels: np.ndarray  # (N,) OrbitClass values
    circulation: np.ndarray  # (N,3) |<L_a>|/<|L_a|> per axis
    tube_axis: np.ndarray  # (N,) index of dominant circulation axis
    planarity: np.ndarray  # (N,) lambda_min/lambda_max of shape tensor
    resonance: np.ndarray  # (N,3) primitive resonance vector (0 if none)
    resonance_order: np.ndarray  # (N,) |n|_1 of the resonance (0 if none)
    class_names: Dict[int, str] = field(default_factory=lambda: dict(CLASS_NAMES))
    radius: Optional[np.ndarray] = None  # (N,) characteristic orbit radius
    ids: Optional[np.ndarray] = None  # (N,) particle IDs

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
                ids=self.ids,
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
            ids=self.ids,
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
            (n_bins + 1,) monotonically increasing radial bin edges (in the
            same units as :attr:`radius`).
        per_bin : bool, optional
            Normalisation of the frequencies. If ``True`` (default), each
            class count in a bin is divided by the number of orbits in that
            bin, so the curves give the class composition within each bin
            (summing to 1 across classes). If ``False``, counts are divided by
            the total number of binned orbits, so the curves give each class's
            share of the whole population.
        radius : array_like, optional
            (N,) per-orbit radius to bin on. Defaults to :attr:`radius` (the
            ``r_mean`` recorded by :func:`classify_orbits`); supply this
            explicitly when the classification carries no radius.
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
                "classification with classify_orbits (which records r_mean)."
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
            ax.plot(centres, frequency, marker="o", label=self.class_names[cls])

        ax.set_xlabel("radius")
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
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_xlabel("orbit class")
        ax.set_ylabel("number of orbits")
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
                f"{name} ({int(size)})",
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
                f"{name} ({int(size)})",
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
    inner_outer_thresh: float = 0.2,
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
    inner_outer_thresh : float, optional
        Long-axis tubes with a relative "hole" ``rho_x_min / rms_perp`` below
        this are labelled *inner*, else *outer* (a convex/non-convex proxy).

    Returns
    -------
    classification : OrbitClassification
        Per-orbit family labels and the diagnostic quantities used to derive
        them (circulation, tube axis, planarity, resonance vector and order).
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
    # Long-axis (x) tubes split inner/outer by relative hole size.
    lat = tube & (tube_axis == 0)
    rms_perp = np.sqrt(np.maximum(c("Syy") + c("Szz"), 1e-30))
    hole = c("rho_x_min") / rms_perp
    labels[lat & (hole < inner_outer_thresh)] = OrbitClass.INNER_LONG_AXIS_TUBE
    labels[lat & (hole >= inner_outer_thresh)] = OrbitClass.OUTER_LONG_AXIS_TUBE

    # Boxes: no circulation. A low-order (>=2) resonance marks a boxlet.
    box = ok & ~is_loop
    labels[box] = OrbitClass.BOX
    labels[box & (res_ord >= 2)] = OrbitClass.BOXLET

    counts = {
        CLASS_NAMES[int(v)]: int(cnt)
        for v, cnt in zip(*np.unique(labels, return_counts=True))
    }
    logger.info("Classified %d orbits: %s", N, counts)

    return OrbitClassification(
        labels=labels,
        circulation=circ,
        tube_axis=tube_axis,
        planarity=planarity,
        resonance=res_vec,
        resonance_order=res_ord,
        radius=c("r_mean"),
        ids=getattr(results, "ids", None),
    )
