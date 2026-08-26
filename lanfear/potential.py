"""High-level potential construction, unit handling, and validation.

:class:`Potential` wraps the C++ :class:`lanfear._core.SCFPotential`. It takes a
:class:`ParticleSystem`, splits off the black hole(s), normalises the field
particles into Hernquist-Ostriker (HO) units, builds the SCF expansion, and
re-attaches each black hole as a softened point mass at its actual (possibly
off-centre) position.

It also provides :meth:`validate`, which compares the analytical SCF potential
to the direct-summation potential of the simulation particles and reports the
fractional agreement -- the ``< X%`` check in the workflow.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from . import _core
from ._logging import get_logger
from .particle_system import ParticleSystem

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Summary of the SCF-vs-direct comparison (dimensionless relative errors).

    Parameters
    ----------
    radii : numpy.ndarray
        HO-unit sample radii at which the comparison was made.
    rel_error : numpy.ndarray
        ``|Phi_scf - Phi_direct| / |Phi_direct|`` per sample point (or per shell).
    median : float
        Median relative error.
    p90 : float
        90th-percentile relative error.
    worst : float
        Maximum relative error.
    """

    radii: np.ndarray  # HO-unit sample radii
    rel_error: np.ndarray  # |Phi_scf - Phi_direct| / |Phi_direct| per point
    median: float
    p90: float
    worst: float

    def passed(self, tolerance: float) -> bool:
        """Test whether the fit meets a tolerance.

        Parameters
        ----------
        tolerance : float
            Maximum acceptable median relative error (a fraction, e.g. ``0.02``).

        Returns
        -------
        passed : bool
            True if the median relative error is below ``tolerance``.
        """
        return self.median < tolerance

    def __repr__(self) -> str:
        """Return a concise string summary of the relative errors.

        Returns
        -------
        text : str
            One-line summary with the median, p90 and worst errors.
        """
        return (
            f"ValidationResult(median={self.median:.2%}, p90={self.p90:.2%}, "
            f"worst={self.worst:.2%}, n={len(self.rel_error)})"
        )


@dataclass
class TruncationSweep:
    """Validation error vs SCF truncation order (a convergence sweep).

    Two one-dimensional sweeps: the radial order ``n_max`` is varied at fixed
    (largest) ``l_max``, and ``l_max`` is varied at fixed (largest) ``n_max``.
    The median SCF-vs-direct error should flatten once the expansion resolves
    the field; continued improvement means the order is still too low.

    Parameters
    ----------
    n_max_values : numpy.ndarray
        Radial orders tested (at ``l_max = l_max_at_n``).
    l_max_at_n : int
        The ``l_max`` used for the ``n_max`` sweep.
    median_error_vs_n : numpy.ndarray
        Median relative error at each ``n_max``.
    l_max_values : numpy.ndarray
        Angular orders tested (at ``n_max = n_max_at_l``).
    n_max_at_l : int
        The ``n_max`` used for the ``l_max`` sweep.
    median_error_vs_l : numpy.ndarray
        Median relative error at each ``l_max``.
    """

    n_max_values: np.ndarray
    l_max_at_n: int
    median_error_vs_n: np.ndarray
    l_max_values: np.ndarray
    n_max_at_l: int
    median_error_vs_l: np.ndarray

    def plot(self, axes=None):
        """Plot the median validation error against ``n_max`` and ``l_max``.

        Parameters
        ----------
        axes : pair of matplotlib.axes.Axes, optional
            The ``(ax_n, ax_l)`` axes to draw into. A new 1x2 figure is created
            if omitted.

        Returns
        -------
        axes : numpy.ndarray of matplotlib.axes.Axes
            The ``(ax_n, ax_l)`` axes drawn on.
        """
        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(9, 4))
        ax_n, ax_l = axes
        ax_n.semilogy(self.n_max_values, self.median_error_vs_n, "o-")
        ax_n.set_xlabel(r"$n_{\max}$")
        ax_n.set_ylabel("median relative error")
        ax_n.set_title(rf"$l_{{\max}} = {self.l_max_at_n}$")
        ax_l.semilogy(self.l_max_values, self.median_error_vs_l, "o-")
        ax_l.set_xlabel(r"$l_{\max}$")
        ax_l.set_ylabel("median relative error")
        ax_l.set_title(rf"$n_{{\max}} = {self.n_max_at_l}$")
        ax_n.figure.tight_layout()
        return np.asarray([ax_n, ax_l], dtype=object)


class Potential:
    """Analytical potential (SCF field + softened point-mass black holes)."""

    # Gravitational constant in the default Gadget unit system
    # (kpc, 1e10 Msun, km/s): G = 4.30091e-6 kpc (km/s)^2 / Msun * 1e10.
    DEFAULT_G = 43009.1

    def __init__(
        self,
        scf: "_core.SCFPotential",
        scale_radius: float,
        field_mass: float,
        field_pos_ho: np.ndarray,
        field_mass_ho: np.ndarray,
        G: float = DEFAULT_G,
    ) -> None:
        """Wrap a built C++ SCF potential and record its unit system.

        Most callers should use :meth:`from_particles` rather than constructing
        this directly.

        Parameters
        ----------
        scf : lanfear._core.SCFPotential
            The built C++ SCF expansion (field particles only).
        scale_radius : float
            Physical length used as the HO length unit.
        field_mass : float
            Total physical mass of the field (non-BH) particles; the HO mass unit.
        field_pos_ho : numpy.ndarray
            (N, 3) field-particle positions in HO units (kept for validation).
        field_mass_ho : numpy.ndarray
            (N,) field-particle masses in HO units (kept for validation).
        G : float, optional
            Gravitational constant in the physical unit system, setting the HO
            velocity/time units. Defaults to the Gadget value.
        """
        self._scf = scf
        self.scale_radius = scale_radius
        self.field_mass = field_mass
        # Physical G sets the HO velocity/time units (G = M_field = a = 1):
        #   V = sqrt(G * M_field / a),  T = a / V.
        self.G = G
        self.velocity_unit = np.sqrt(G * field_mass / scale_radius)
        self.time_unit = scale_radius / self.velocity_unit
        # Retained (in HO units) for the direct-summation validation.
        self._field_pos_ho = field_pos_ho
        self._field_mass_ho = field_mass_ho
        # Black-hole parameters in HO units, mirrored from the C++ side so we
        # can isolate the field contribution during validation.
        self._bh_params: list = []  # each: (mass_ho, np.array([x,y,z]), soft)

    @property
    def core(self):
        """The underlying picklable C++ potential (for the orbit drivers).

        Returns
        -------
        core : lanfear._core.SCFPotential
            The wrapped C++ potential object.
        """
        return self._scf

    # ------------------------------------------------------------- builders
    @classmethod
    def from_particles(
        cls,
        particles: ParticleSystem,
        n_max: int,
        l_max: int,
        bh_softening: float = 1e-3,
        G: float = DEFAULT_G,
    ) -> "Potential":
        """Build the analytical potential from a prepared ParticleSystem.

        Parameters
        ----------
        particles : ParticleSystem
            A system that has already been :meth:`~ParticleSystem.prepare`\\ d
            (recentred, aligned, scale radius estimated).
        n_max : int
            Radial truncation order of the HO expansion.
        l_max : int
            Angular (spherical-harmonic) truncation order of the HO expansion.
        bh_softening : float, optional
            Plummer softening for each black hole, in units of the scale radius.
        G : float, optional
            Gravitational constant in the physical unit system (default Gadget).

        Returns
        -------
        potential : Potential
            The analytical potential with any black holes re-attached.
        """
        if particles.scale_radius is None:
            particles.estimate_scale_radius()
        a = particles.scale_radius

        field = particles.field
        field_mass = float(np.sum(field.mass))
        pos_ho = field.pos / a
        mass_ho = field.mass / field_mass

        t0 = time.perf_counter()
        scf = _core.SCFPotential(n_max, l_max, pos_ho, mass_ho)
        elapsed = time.perf_counter() - t0
        logger.info(
            f"Built HO SCF potential (n_max={n_max}, l_max={l_max}) from "
            f"{field.n_particles} field particles in {elapsed:.2f} s"
        )

        pot = cls(scf, a, field_mass, pos_ho, mass_ho, G=G)

        # Re-attach black holes at their true positions (HO units).
        bh = particles.black_holes
        for i in range(bh.n_particles):
            pot.add_black_hole(
                mass=float(bh.mass[i]),
                position=bh.pos[i],
                softening=bh_softening,
            )
        if bh.n_particles:
            logger.info(f"Attached {bh.n_particles} black hole(s) to the potential")
        return pot

    def add_black_hole(self, mass, position, softening: float = 1e-3) -> None:
        """Add a softened point mass to the potential.

        Parameters
        ----------
        mass : float
            Black-hole mass in *physical* units.
        position : array-like of float
            (3,) black-hole position in *physical* units.
        softening : float, optional
            Plummer softening in scale-radius (HO) units.
        """
        pos_ho = np.asarray(position, dtype=np.float64) / self.scale_radius
        mass_ho = mass / self.field_mass
        self._scf.add_black_hole(
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
    @property
    def n_black_holes(self) -> int:
        """Number of black holes attached to the potential.

        Returns
        -------
        n : int
            The number of softened point masses.
        """
        return self._scf.num_black_holes

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
        return self._scf.potential_batch(pts)

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
        return self._scf.acceleration_batch(pts)

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
        return self._scf.potential_batch(np.ascontiguousarray(points_ho))

    # ---------------------------------------------------------- validation
    def _direct_potential_ho(
        self, points_ho: np.ndarray, softening: float
    ) -> np.ndarray:
        """Direct-summation potential of the field particles (HO units).

        This is the "actual" simulation potential the SCF fit is checked
        against. An O(n_points * n_field) brute-force sum, computed in the C++
        core (OpenMP-parallel over evaluation points) rather than in Python --
        see ``_core.direct_potential_batch``.

        Parameters
        ----------
        points_ho : numpy.ndarray
            (N, 3) evaluation points in HO units.
        softening : float
            Plummer softening (HO units) applied to the direct sum.

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

    def validate(
        self,
        n_shells: int = 24,
        n_directions: int = 64,
        r_range: Optional[tuple] = None,
        softening: Optional[float] = None,
        include_bh: bool = False,
        seed: int = 0,
    ) -> ValidationResult:
        """Compare the SCF potential against direct summation.

        Samples ``n_directions`` random directions on each of ``n_shells``
        log-spaced radii, averaging away particle discreteness noise, and
        reports the fractional agreement.

        Parameters
        ----------
        n_shells : int, optional
            Number of log-spaced radial shells to sample.
        n_directions : int, optional
            Number of isotropic directions sampled per shell.
        r_range : tuple of float, optional
            ``(r_min, r_max)`` in *physical* units. Defaults to the 5th-95th
            percentile of the field-particle radii.
        softening : float, optional
            Softening for the direct sum, in HO units. Defaults to a mean
            interparticle-spacing estimate, which tames discreteness noise.
        include_bh : bool, optional
            If True the SCF side includes the black-hole term. Left off by
            default so the check targets the *field* expansion, which is what the
            SCF is responsible for representing.
        seed : int, optional
            Seed for the random direction sampling.

        Returns
        -------
        result : ValidationResult
            The per-shell and aggregate relative errors.
        """
        rng = np.random.default_rng(seed)
        r_field = np.linalg.norm(self._field_pos_ho, axis=1)
        if r_range is None:
            r_min, r_max = np.percentile(r_field, [5, 95])
        else:
            r_min, r_max = np.asarray(r_range) / self.scale_radius
        radii = np.logspace(np.log10(r_min), np.log10(r_max), n_shells)

        if softening is None:
            # Local mean interparticle spacing at the *innermost* sampled
            # radius. Basing this on r_min (not r_max) keeps the softening small
            # where the density is high, so it does not bias the direct-sum
            # reference low in the inner region. The residual there is then set
            # by Poisson noise in the enclosed mass (~1/sqrt(N(<r))), not by
            # softening.
            n_in = max(1, np.sum(r_field < r_min))
            softening = float(r_min / n_in ** (1.0 / 3.0))

        # Build sample points: for each shell, n_directions isotropic points.
        mu = rng.uniform(-1, 1, (n_shells, n_directions))
        az = rng.uniform(0, 2 * np.pi, (n_shells, n_directions))
        st = np.sqrt(1 - mu**2)
        dirs = np.stack(
            [st * np.cos(az), st * np.sin(az), mu], axis=-1
        )  # (n_shells, n_directions, 3)
        pts = (radii[:, None, None] * dirs).reshape(-1, 3)

        phi_scf = self._potential_ho(pts)
        if not include_bh and self.n_black_holes > 0:
            # Subtract the point-mass contribution so we compare field-to-field.
            phi_scf = phi_scf - self._bh_potential_ho(pts)
        phi_direct = self._direct_potential_ho(pts, softening)

        rel = np.abs(phi_scf - phi_direct) / np.abs(phi_direct)
        # Aggregate per shell (median over directions) for a clean radial curve.
        rel_shell = np.median(rel.reshape(n_shells, n_directions), axis=1)
        result = ValidationResult(
            radii=radii,
            rel_error=rel_shell,
            median=float(np.median(rel)),
            p90=float(np.percentile(rel, 90)),
            worst=float(np.max(rel)),
        )
        logger.info(
            f"SCF validation vs direct sum: median={100 * result.median:.2f}% "
            f"p90={100 * result.p90:.2f}% worst={100 * result.worst:.2f}%"
        )
        return result

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
        the left panel is the fitted potential (the SCF field expansion plus
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
            Plummer softening (scale-radius/HO units) applied to the direct-
            summation "true" potential (default 1e-3, matching the default
            black-hole softening).
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

    @classmethod
    def truncation_convergence(
        cls,
        particles: ParticleSystem,
        n_max_values,
        l_max_values,
        G: float = DEFAULT_G,
        **validate_kwargs,
    ) -> TruncationSweep:
        """Validation-error-vs-truncation sweep (rebuilds the expansion).

        Rebuilds the SCF potential over a range of truncation orders and records
        the median SCF-vs-direct error, so convergence in ``n_max`` and ``l_max``
        can be read off directly. ``n_max`` is swept at the largest ``l_max`` and
        vice versa. This is more expensive than
        :meth:`coefficient_power_spectrum` (each point is a full build plus
        :meth:`validate`), so use a modest grid (and/or a particle subsample).

        Parameters
        ----------
        particles : ParticleSystem
            The (prepared) system to rebuild the expansion from.
        n_max_values : sequence of int
            Radial orders to test.
        l_max_values : sequence of int
            Angular orders to test.
        G : float, optional
            Gravitational constant in the physical unit system.
        **validate_kwargs
            Passed to :meth:`validate` (e.g. ``n_shells``, ``n_directions``,
            ``seed``).

        Returns
        -------
        sweep : TruncationSweep
            The median error along each of the two truncation sweeps.
        """
        n_values = np.asarray(sorted(int(v) for v in n_max_values))
        l_values = np.asarray(sorted(int(v) for v in l_max_values))
        l_ref = int(l_values[-1])
        n_ref = int(n_values[-1])

        logger.info(
            f"Truncation sweep: n_max in {list(n_values)} (l_max={l_ref}), "
            f"l_max in {list(l_values)} (n_max={n_ref})"
        )
        err_vs_n = np.array(
            [
                cls.from_particles(particles, n_max=int(nm), l_max=l_ref, G=G)
                .validate(**validate_kwargs)
                .median
                for nm in n_values
            ]
        )
        err_vs_l = np.array(
            [
                cls.from_particles(particles, n_max=n_ref, l_max=int(lm), G=G)
                .validate(**validate_kwargs)
                .median
                for lm in l_values
            ]
        )
        return TruncationSweep(
            n_max_values=n_values,
            l_max_at_n=l_ref,
            median_error_vs_n=err_vs_n,
            l_max_values=l_values,
            n_max_at_l=n_ref,
            median_error_vs_l=err_vs_l,
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
