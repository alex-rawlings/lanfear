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
from ._potential_base import _PotentialBase
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


class Potential(_PotentialBase):
    """Analytical potential (SCF field + softened point-mass black holes)."""

    DEFAULT_G = _PotentialBase.DEFAULT_G

    def __init__(
        self,
        scf: "_core.SCFPotential",
        scale_radius: float,
        field_mass: float,
        field_pos_ho: np.ndarray,
        field_mass_ho: np.ndarray,
        G: float = DEFAULT_G,
        n_max: Optional[int] = None,
        l_max: Optional[int] = None,
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
        n_max : int, optional
            Radial truncation order the expansion was built with (recorded for
            provenance; see :meth:`from_particles`).
        l_max : int, optional
            Angular truncation order the expansion was built with (recorded for
            provenance; see :meth:`from_particles`).
        """
        self._core = scf
        self._set_units(scale_radius, field_mass, G)
        self.n_max = n_max
        self.l_max = l_max
        # Retained (in HO units) for the direct-summation validation.
        self._field_pos_ho = field_pos_ho
        self._field_mass_ho = field_mass_ho

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
            Spline (Gadget4) softening length for each black hole, in units of
            the scale radius.
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

        pot = cls(scf, a, field_mass, pos_ho, mass_ho, G=G, n_max=n_max, l_max=l_max)

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

    # ---------------------------------------------------------- validation
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
            r_min, r_max = np.percentile(r_field, [1, 99])
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
            softening = self._default_softening(r_min)

        # Build sample points: for each shell, n_directions isotropic points.
        mu = rng.uniform(-1, 1, (n_shells, n_directions))
        az = rng.uniform(0, 2 * np.pi, (n_shells, n_directions))
        st = np.sqrt(1 - mu**2)
        dirs = np.stack(
            [st * np.cos(az), st * np.sin(az), mu], axis=-1
        )  # (n_shells, n_directions, 3)
        pts = (radii[:, None, None] * dirs).reshape(-1, 3)

        return self._validate_points(
            pts, radii, softening, include_bh, group_size=n_directions
        )

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
