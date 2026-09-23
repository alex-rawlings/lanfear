"""Shared unit handling, black-hole bookkeeping, validation and plotting for
every potential class (:class:`~lanfear.Potential`, :class:`~lanfear.DiscPotential`,
:class:`~lanfear.MultiComponentPotential`).

A subclass sets ``self._core`` (the picklable C++ potential -- an
``SCFPotential``, ``DiscPotential`` or ``CompositePotential`` -- exposing
``potential_batch``/``acceleration_batch``/``add_black_hole``/``num_black_holes``),
calls :meth:`_PotentialBase._set_units` with its length/mass unit, and sets
``self._field_pos_ho``/``self._field_mass_ho`` (the field particles in its own
HO units, kept for the direct-summation goodness-of-fit check). Everything
else -- unit conversions, black holes, evaluation, validation plumbing and the
potential-plane plot -- is implemented once, here.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from . import _core
from ._logging import get_logger

logger = get_logger(__name__)


class _PotentialBase:
    """Mixin providing the API shared by every potential class.

    Not instantiated directly; see :class:`~lanfear.Potential`,
    :class:`~lanfear.DiscPotential` and :class:`~lanfear.MultiComponentPotential`.
    """

    # Gravitational constant in the default Gadget unit system
    # (kpc, 1e10 Msun, km/s): G = 4.30091e-6 kpc (km/s)^2 / Msun * 1e10.
    DEFAULT_G = 43009.1

    # Truncation order attributes read by lanfear.orbits.analyse_family for
    # OrbitResults provenance. Only meaningful for a single SCF component
    # (Potential overrides both with real ints); None elsewhere (DiscPotential,
    # MultiComponentPotential) is the correct "not a single truncation order"
    # answer, and OrbitResults.n_max/l_max are already Optional.
    n_max: Optional[int] = None
    l_max: Optional[int] = None

    def _set_units(self, scale_radius: float, field_mass: float, G: float) -> None:
        """Derive and store the length/mass/velocity/time unit system.

        Parameters
        ----------
        scale_radius : float
            Physical length used as this potential's HO length unit.
        field_mass : float
            Physical mass used as this potential's HO mass unit.
        G : float
            Gravitational constant in the physical unit system.
        """
        self.scale_radius = scale_radius
        self.field_mass = field_mass
        self.G = G
        # Physical G sets the HO velocity/time units (G = M = a = 1):
        #   V = sqrt(G * M / a),  T = a / V.
        self.velocity_unit = np.sqrt(G * field_mass / scale_radius)
        self.time_unit = scale_radius / self.velocity_unit
        self._bh_params: list = []  # each: (mass_ho, np.array([x,y,z]), soft)

    @property
    def core(self):
        """The underlying picklable C++ potential (for the orbit drivers).

        Returns
        -------
        core : lanfear._core.SCFPotential, DiscPotential or CompositePotential
            The wrapped C++ potential object.
        """
        return self._core

    # ---------------------------------------------------------- black holes
    @property
    def n_black_holes(self) -> int:
        """Number of black holes attached to the potential.

        Returns
        -------
        n : int
            The number of softened point masses.
        """
        return self._core.num_black_holes

    def add_black_hole(self, mass, position, softening: float = 1e-3) -> None:
        """Add a softened point mass to the potential.

        Parameters
        ----------
        mass : float
            Black-hole mass in *physical* units.
        position : array-like of float
            (3,) black-hole position in *physical* units.
        softening : float, optional
            Spline (Gadget4) softening length in scale-radius (HO) units.
        """
        pos_ho = np.asarray(position, dtype=np.float64) / self.scale_radius
        mass_ho = mass / self.field_mass
        self._core.add_black_hole(
            mass_ho,
            float(pos_ho[0]),
            float(pos_ho[1]),
            float(pos_ho[2]),
            softening,
        )
        self._bh_params.append((mass_ho, pos_ho, softening))
        logger.debug(
            f"Added black hole: mass_ho={mass_ho:.3g} "
            f"pos_ho={np.round(pos_ho, 4)} softening={softening:.3g}"
        )

    def _bh_potential_ho(self, points_ho: np.ndarray) -> np.ndarray:
        """Black-hole-only potential (HO units), for isolating the field.

        Parameters
        ----------
        points_ho : numpy.ndarray
            (N, 3) evaluation points in HO units.

        Returns
        -------
        phi : numpy.ndarray
            (N,) summed softened point-mass potential of the black holes.
        """
        out = np.zeros(len(points_ho))
        for mass_ho, pos_ho, soft in self._bh_params:
            d = points_ho - pos_ho[None, :]
            r = np.sqrt(np.einsum("ij,ij->i", d, d) + soft * soft)
            out += -mass_ho / r
        return out

    # ------------------------------------------------------------- units
    def to_ho_state(self, pos_phys: np.ndarray, vel_phys: np.ndarray) -> np.ndarray:
        """Convert physical positions/velocities to HO integration states.

        Parameters
        ----------
        pos_phys : numpy.ndarray
            (N, 3) positions in physical length units.
        vel_phys : numpy.ndarray
            (N, 3) velocities in physical velocity units (km/s for the default
            Gadget system).

        Returns
        -------
        states : numpy.ndarray
            (N, 6) ``(x, y, z, vx, vy, vz)`` states in HO units, ready for
            ``integrate_batch``/``analyse_batch``.
        """
        pos = np.atleast_2d(np.asarray(pos_phys, dtype=np.float64))
        vel = np.atleast_2d(np.asarray(vel_phys, dtype=np.float64))
        states = np.empty((len(pos), 6))
        states[:, :3] = pos / self.scale_radius
        states[:, 3:] = vel / self.velocity_unit
        return states

    def period_to_physical(self, period_ho: np.ndarray) -> np.ndarray:
        """Convert an HO-unit period/time to physical time units.

        Parameters
        ----------
        period_ho : array-like of float
            Time(s) in HO units.

        Returns
        -------
        period_phys : numpy.ndarray
            The same time(s) in physical units.
        """
        return np.asarray(period_ho) * self.time_unit

    # ---------------------------------------------------------- evaluation
    def potential(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the potential at physical Cartesian points.

        Parameters
        ----------
        points : numpy.ndarray
            (N, 3) points in physical length units.

        Returns
        -------
        phi : numpy.ndarray
            (N,) potential values in HO units.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64)) / self.scale_radius
        return self._core.potential_batch(pts)

    def acceleration(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the acceleration at physical Cartesian points.

        Parameters
        ----------
        points : numpy.ndarray
            (N, 3) points in physical length units.

        Returns
        -------
        acc : numpy.ndarray
            (N, 3) accelerations in HO units.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64)) / self.scale_radius
        return self._core.acceleration_batch(pts)

    def _potential_ho(self, points_ho: np.ndarray) -> np.ndarray:
        """Evaluate the potential at points already given in HO units.

        Parameters
        ----------
        points_ho : numpy.ndarray
            (N, 3) points in HO units.

        Returns
        -------
        phi : numpy.ndarray
            (N,) potential values in HO units.
        """
        return self._core.potential_batch(np.ascontiguousarray(points_ho))

    # ---------------------------------------------------------- validation
    def _direct_potential_ho(
        self, points_ho: np.ndarray, softening: float
    ) -> np.ndarray:
        """Direct-summation potential of the field particles (HO units).

        This is the "actual" simulation potential the analytic fit is checked
        against. An O(n_points * n_field) brute-force sum, computed in the C++
        core (OpenMP-parallel over evaluation points) rather than in Python --
        see ``_core.direct_potential_batch``.

        Parameters
        ----------
        points_ho : numpy.ndarray
            (N, 3) evaluation points in HO units.
        softening : float
            Spline (Gadget4) softening length (HO units) applied to the direct sum.

        Returns
        -------
        phi : numpy.ndarray
            (N,) direct-summation potential in HO units.
        """
        t0 = time.perf_counter()
        phi = _core.direct_potential_batch(
            np.ascontiguousarray(points_ho, dtype=np.float64),
            self._field_pos_ho,
            self._field_mass_ho,
            softening,
        )
        elapsed = time.perf_counter() - t0
        logger.info(f"True potential calculated in {elapsed:.2f} s")
        return phi

    def _default_softening(self, r_reference_ho: float) -> float:
        """Mean interparticle-spacing softening estimate (HO units).

        Used as the default direct-summation softening in :meth:`validate`
        implementations: a local estimate ``r / N(<r)^(1/3)`` at a reference
        radius, which tames Poisson noise without biasing the reference low.

        Parameters
        ----------
        r_reference_ho : float
            Reference radius (HO units) at which to estimate the local
            interparticle spacing.

        Returns
        -------
        softening : float
            The estimated softening length (HO units).
        """
        r_field = np.linalg.norm(self._field_pos_ho, axis=1)
        n_in = max(1, int(np.sum(r_field < r_reference_ho)))
        return float(r_reference_ho / n_in ** (1.0 / 3.0))

    def _validate_points(
        self,
        pts_ho: np.ndarray,
        radii: np.ndarray,
        softening: float,
        include_bh: bool,
        group_size: Optional[int] = None,
    ):
        """Shared tail of every ``validate()``: evaluate, compare, summarise.

        Subclasses build ``pts_ho`` however suits their geometry (isotropic
        shells for :class:`~lanfear.Potential`, field-particle probes for
        :class:`~lanfear.DiscPotential` and :class:`~lanfear.MultiComponentPotential`)
        and hand the points to this method, which does the actual fit-vs-direct-sum
        comparison and builds the :class:`~lanfear.ValidationResult`.

        Parameters
        ----------
        pts_ho : numpy.ndarray
            (n_points, 3) evaluation points in HO units (``n_points =
            len(radii) * group_size`` when ``group_size`` is given, else
            ``len(radii)``).
        radii : numpy.ndarray
            Radii reported on the result: one row per shell when
            ``group_size`` is given, otherwise one per point.
        softening : float
            Direct-summation softening (HO units).
        include_bh : bool
            If True the analytic side includes the black-hole term.
        group_size : int, optional
            When given, ``pts_ho`` is treated as ``len(radii)`` groups of this
            many points each (e.g. isotropic directions per shell), and the
            reported per-``radii`` error is the median over each group.
            Omit for one point per radius.

        Returns
        -------
        result : lanfear.ValidationResult
            The per-``radii`` and aggregate relative errors.
        """
        from .potential import ValidationResult

        phi_fit = self._potential_ho(pts_ho)
        if not include_bh and self.n_black_holes > 0:
            phi_fit = phi_fit - self._bh_potential_ho(pts_ho)
        phi_direct = self._direct_potential_ho(pts_ho, softening)

        rel = np.abs(phi_fit - phi_direct) / np.abs(phi_direct)
        if group_size is not None:
            rel_report = np.median(rel.reshape(len(radii), group_size), axis=1)
        else:
            rel_report = rel
        result = ValidationResult(
            radii=radii,
            rel_error=rel_report,
            median=float(np.median(rel)),
            p90=float(np.percentile(rel, 90)),
            worst=float(np.max(rel)),
        )
        logger.info(
            f"Validation vs direct sum: median={100 * result.median:.2f}% "
            f"p90={100 * result.p90:.2f}% worst={100 * result.worst:.2f}%"
        )
        return result

    # ------------------------------------------------------------ plotting
    def plot_potential_plane(
        self,
        centre,
        box_size,
        plane: str = "xy",
        n_grid: int = 150,
        softening: float = 1e-3,
        axes=None,
        cmap: str = "bone",
        residual_cmap: str = "RdBu_r",
    ):
        """Filled-contour plot of the potential in a thin planar slice.

        Evaluates the potential on a regular grid spanning ``box_size`` about
        ``centre``, in the requested coordinate plane, at zero thickness (a
        true 2-D slice with the third coordinate held fixed at ``centre``'s
        component, not a projection or column sum). Produces a 1x2 figure:
        the left panel is the fitted potential (the analytic expansion plus
        any black holes, i.e. what :meth:`potential` returns); the right
        panel is the residual ``fitted - true``, where "true" is the direct-
        summation potential of the field particles plus the same analytic
        black-hole term(s) (see :meth:`validate`).

        Parameters
        ----------
        centre : array-like of float
            (3,) physical-unit centre of the slice.
        box_size : tuple of float
            ``(length_1, length_2)`` physical-unit side lengths of the box
            along the plane's two in-plane axes.
        plane : {"xy", "xz", "yz"}, optional
            Coordinate plane to slice (default ``"xy"``); the third
            coordinate is held fixed at the corresponding component of
            ``centre``.
        n_grid : int, optional
            Number of grid points per side (default 150).
        softening : float, optional
            Spline (Gadget4) softening length (scale-radius/HO units) applied
            to the direct-summation "true" potential (default 1e-3, matching
            the default black-hole softening).
        axes : pair of matplotlib.axes.Axes, optional
            The ``(ax_fit, ax_residual)`` axes to draw into. A new 1x2 figure
            is created if omitted.
        cmap : str, optional
            Colormap for the fitted-potential panel.
        residual_cmap : str, optional
            Diverging colormap for the residual panel (centred on zero).

        Returns
        -------
        axes : numpy.ndarray of matplotlib.axes.Axes
            The ``(ax_fit, ax_residual)`` axes drawn on.

        Raises
        ------
        ValueError
            If ``plane`` is not one of ``"xy"``, ``"xz"``, ``"yz"``.
        """
        axis_indices = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}
        if plane not in axis_indices:
            raise ValueError(
                f"plane must be one of {sorted(axis_indices)}, got '{plane}'"
            )
        i, j, k = axis_indices[plane]

        centre = np.asarray(centre, dtype=np.float64)
        length_1, length_2 = box_size
        u = np.linspace(-0.5 * length_1, 0.5 * length_1, n_grid) + centre[i]
        v = np.linspace(-0.5 * length_2, 0.5 * length_2, n_grid) + centre[j]
        grid_u, grid_v = np.meshgrid(u, v)

        points = np.empty((grid_u.size, 3))
        points[:, i] = grid_u.ravel()
        points[:, j] = grid_v.ravel()
        points[:, k] = centre[k]

        phi_fit = self.potential(points).reshape(grid_u.shape)

        points_ho = points / self.scale_radius
        phi_true_ho = self._direct_potential_ho(points_ho, softening)
        if self.n_black_holes:
            phi_true_ho = phi_true_ho + self._bh_potential_ho(points_ho)
        phi_true = phi_true_ho.reshape(grid_u.shape)

        residual = (phi_fit - phi_true) / phi_true

        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax_fit, ax_res = axes

        cf = ax_fit.contourf(grid_u, grid_v, phi_fit, levels=32, cmap=cmap)
        ax_fit.figure.colorbar(cf, ax=ax_fit, label=r"$\Phi_{\rm fit}$ (HO units)")
        ax_fit.set_title("Fitted potential")

        lim = float(np.max(np.abs(residual))) or 1e-12
        levels_res = np.linspace(-lim, lim, 33)
        rf = ax_res.contourf(
            grid_u, grid_v, residual, levels=levels_res, cmap=residual_cmap
        )
        ax_res.figure.colorbar(
            rf,
            ax=ax_res,
            label=r"$(\Phi_{\rm fit} - \Phi_{\rm true}) / \Phi_{\rm true}$",
        )
        ax_res.set_title("Relative residual")

        xlabel, ylabel = plane[0], plane[1]
        for ax in (ax_fit, ax_res):
            ax.set_xlabel(rf"${xlabel}$")
            ax.set_ylabel(rf"${ylabel}$")
            ax.set_aspect("equal")
            ax.grid(False)
        ax_fit.figure.tight_layout()
        return np.asarray([ax_fit, ax_res], dtype=object)
