"""Disc-adapted basis-function potential for flattened systems.

Wraps :class:`lanfear._core.DiscPotential` (a Miyamoto-Nagai basis). The field
particles are expanded in a set of MN density-potential pairs of several radial
and vertical scales; the (non-orthonormal) coefficients solve the Galerkin
system ``gram @ c = b`` where ``b`` is the SCF particle sum and ``gram`` is the
fixed basis Gram matrix. The black hole is re-attached as a softened point mass,
exactly as for the spheroidal :class:`~lanfear.Potential`.

Use this for strongly flattened / disc-like systems; use :class:`~lanfear.Potential`
(Hernquist-Ostriker) for spherical-ish ones.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from . import _core
from ._logging import get_logger
from ._potential_base import _PotentialBase
from .particle_system import ParticleSystem

if TYPE_CHECKING:
    from .potential import ValidationResult

logger = get_logger(__name__)


# NumPy Miyamoto-Nagai primitives, matching lanfear/disc_potential.hpp exactly
# (unit-tested against the C++ versions in tests/test_disc.py).
def _mn_potential(x, y, z, a, b):
    """Potential of a unit-mass Miyamoto-Nagai component (vectorised).

    Parameters
    ----------
    x, y, z : float or numpy.ndarray
        Cartesian coordinates in HO units; broadcast together.
    a : float
        Radial scale length.
    b : float
        Vertical scale length.

    Returns
    -------
    phi : float or numpy.ndarray
        Potential of a unit-mass MN component.
    """
    zeta = np.sqrt(z * z + b * b)
    return -1.0 / np.sqrt(x * x + y * y + (a + zeta) ** 2)


def _mn_density(R, z, a, b):
    """Density of a unit-mass Miyamoto-Nagai component (vectorised).

    Parameters
    ----------
    R : float or numpy.ndarray
        Cylindrical radius ``sqrt(x^2 + y^2)`` in HO units.
    z : float or numpy.ndarray
        Height above the plane in HO units.
    a : float
        Radial scale length.
    b : float
        Vertical scale length.

    Returns
    -------
    rho : float or numpy.ndarray
        Density of a unit-mass MN component.
    """
    zeta = np.sqrt(z * z + b * b)
    az = a + zeta
    num = b * b * (a * R * R + (a + 3.0 * zeta) * az * az)
    den = 4.0 * np.pi * (R * R + az * az) ** 2.5 * zeta**3
    return num / den


class DiscPotential(_PotentialBase):
    """Analytical disc potential (MN basis field + softened black holes)."""

    DEFAULT_G = _PotentialBase.DEFAULT_G

    def __init__(
        self,
        core,
        scale_radius,
        field_mass,
        field_pos_ho,
        field_mass_ho,
        gram,
        G=DEFAULT_G,
    ):
        """Wrap a built C++ disc potential and record its unit system.

        Most callers should use :meth:`from_particles` rather than constructing
        this directly.

        Parameters
        ----------
        core : lanfear._core.DiscPotential
            The built C++ disc-basis potential (coefficients already set).
        scale_radius : float
            Physical length used as the HO length unit.
        field_mass : float
            Total physical mass of the field particles; the HO mass unit.
        field_pos_ho : numpy.ndarray
            (N, 3) field-particle positions in HO units (kept for validation).
        field_mass_ho : numpy.ndarray
            (N,) field-particle masses in HO units (kept for validation).
        gram : numpy.ndarray
            (M, M) basis Gram matrix used to solve for the coefficients.
        G : float, optional
            Gravitational constant in the physical unit system (default Gadget).
        """
        self._core = core
        self._set_units(scale_radius, field_mass, G)
        self.gram = gram
        self._field_pos_ho = field_pos_ho
        self._field_mass_ho = field_mass_ho

    # ------------------------------------------------------------- basis
    @staticmethod
    def default_basis(pos_ho, n_radial=8, n_vert=3) -> Tuple[np.ndarray, np.ndarray]:
        """Build a geometric grid of MN (radial, thickness) scales.

        Radial scales span roughly the cylindrical extent of the particles;
        thickness scales span the vertical extent. Every combination of a radial
        and a vertical scale becomes one basis member.

        Parameters
        ----------
        pos_ho : numpy.ndarray
            (N, 3) field positions in HO units.
        n_radial : int, optional
            Number of radial (``a``) scales.
        n_vert : int, optional
            Number of vertical (``b``) scales.

        Returns
        -------
        a : numpy.ndarray
            (n_radial * n_vert,) radial scale of each basis member.
        b : numpy.ndarray
            (n_radial * n_vert,) vertical scale of each basis member.
        """
        R = np.hypot(pos_ho[:, 0], pos_ho[:, 1])
        z = np.abs(pos_ho[:, 2])
        r_hi = np.percentile(R, 90) + 1e-3
        a_scales = np.geomspace(0.03 * r_hi, 1.5 * r_hi, n_radial)
        z_hi = np.percentile(z, 90) + 1e-3
        b_scales = np.geomspace(0.3 * z_hi, 1.5 * z_hi, n_vert)
        a_list, b_list = [], []
        for a in a_scales:
            for b in b_scales:
                a_list.append(a)
                b_list.append(b)
        return np.array(a_list), np.array(b_list)

    @staticmethod
    def _gram_matrix(a_list, b_list, n_quad=500) -> np.ndarray:
        """Gram matrix ``gram[i, j] = integral rho_i Phi_j dV``.

        Uses log-spaced nodes in R and z (midpoint rule in ln R, ln z) so that
        every basis scale -- which span more than two decades -- is resolved with
        roughly equal accuracy. The MN density is z-symmetric, so z runs over the
        positive half and the result is doubled. Contributions from R, z below the
        inner cutoff are negligible because dV ~ R -> 0 there.

        Parameters
        ----------
        a_list : array-like of float
            Radial scales of the basis members.
        b_list : array-like of float
            Vertical scales of the basis members.
        n_quad : int, optional
            Number of log-spaced quadrature nodes per dimension.

        Returns
        -------
        gram : numpy.ndarray
            (M, M) symmetric Gram matrix.
        """
        M = len(a_list)
        R = np.geomspace(1e-4, 1e3, n_quad)
        z = np.geomspace(1e-4, 1e3, n_quad)
        dlnR = np.log(R[1] / R[0])
        dlnz = np.log(z[1] / z[0])
        RR, ZZ = np.meshgrid(R, z, indexing="ij")
        # dV = 2 * (2 pi R) dR dz, with dR = R dlnR, dz = z dlnz.
        dV = 2.0 * (2.0 * np.pi * RR) * (RR * dlnR) * (ZZ * dlnz)

        rho = [_mn_density(RR, ZZ, a_list[i], b_list[i]) for i in range(M)]
        phi = [_mn_potential(RR, 0.0, ZZ, a_list[j], b_list[j]) for j in range(M)]
        gram = np.empty((M, M))
        for i in range(M):
            for j in range(M):
                gram[i, j] = np.sum(dV * rho[i] * phi[j])
        return 0.5 * (gram + gram.T)  # symmetrise (reciprocity)

    @staticmethod
    def _solve(gram, b, rcond) -> np.ndarray:
        """Solve ``gram @ c = b`` via a truncated pseudo-inverse.

        The MN basis is highly collinear, so the Gram matrix is severely
        ill-conditioned. ``-gram`` is symmetric (positive definite in exact
        arithmetic), so we eigendecompose it and drop modes whose eigenvalue is
        below ``rcond * lambda_max`` -- keeping only the well-determined
        combinations of basis functions. This yields a stable, smooth fit.

        Parameters
        ----------
        gram : numpy.ndarray
            (M, M) basis Gram matrix.
        b : numpy.ndarray
            (M,) SCF particle sum (the right-hand side).
        rcond : float
            Relative eigenvalue-truncation threshold.

        Returns
        -------
        c : numpy.ndarray
            (M,) basis coefficients.
        """
        neg = -0.5 * (gram + gram.T)  # symmetric
        w, V = np.linalg.eigh(neg)
        thresh = rcond * max(w.max(), 1e-300)
        inv_w = np.where(w > thresh, 1.0 / w, 0.0)
        return V @ (inv_w * (V.T @ (-b)))

    # ------------------------------------------------------------- builder
    @classmethod
    def from_particles(
        cls,
        particles: ParticleSystem,
        n_radial: int = 8,
        n_vert: int = 3,
        a_scales=None,
        b_scales=None,
        rcond: float = 1e-4,
        bh_softening: float = 1e-3,
        G: float = DEFAULT_G,
    ) -> "DiscPotential":
        """Build the disc potential from a prepared ParticleSystem.

        Parameters
        ----------
        particles : ParticleSystem
            A system that has already been :meth:`~ParticleSystem.prepare`\\ d
            (recentred, aligned, scale radius estimated).
        n_radial : int, optional
            Number of radial scales in the default basis.
        n_vert : int, optional
            Number of vertical scales in the default basis.
        a_scales, b_scales : array-like of float, optional
            Explicit basis scales; if both are given they override the default
            basis built from the particle distribution.
        rcond : float, optional
            Eigenvalue-truncation level of the (ill-conditioned) Gram solve:
            smaller keeps more basis modes (better fit, less stable), larger is
            smoother/more robust.
        bh_softening : float, optional
            Spline (Gadget4) softening length for each black hole, in
            scale-radius units.
        G : float, optional
            Gravitational constant in the physical unit system (default Gadget).

        Returns
        -------
        potential : DiscPotential
            The disc potential with any black holes re-attached.
        """
        if particles.scale_radius is None:
            particles.estimate_scale_radius()
        a_unit = particles.scale_radius

        field = particles.field
        field_mass = float(np.sum(field.mass))
        pos_ho = field.pos / a_unit
        mass_ho = field.mass / field_mass

        if a_scales is None or b_scales is None:
            a_arr, b_arr = cls.default_basis(pos_ho, n_radial, n_vert)
        else:
            a_arr, b_arr = np.asarray(a_scales, float), np.asarray(b_scales, float)

        t0 = time.perf_counter()
        core = _core.DiscPotential(a_arr, b_arr)
        b = np.asarray(core.scf_sum(pos_ho, mass_ho))
        gram = cls._gram_matrix(a_arr, b_arr)
        c = cls._solve(gram, b, rcond)
        core.set_coefficients(np.ascontiguousarray(c))
        elapsed = time.perf_counter() - t0
        logger.info(
            f"Built disc potential ({len(a_arr)} Miyamoto-Nagai basis functions) "
            f"from {field.n_particles} field particles in {elapsed:.2f} s"
        )
        c_sum = float(np.sum(c))
        cond = np.linalg.cond(gram)
        logger.debug(
            f"Gram condition number {cond:.2e}; sum(coefficients)={c_sum:.3f} "
            f"(monopole ~ 1)"
        )
        if not (0.8 < c_sum < 1.2):
            logger.warning(
                f"Disc coefficient sum {c_sum:.3f} is far from 1; the monopole "
                f"(total mass) may be poorly represented -- consider more basis "
                f"functions or a different rcond."
            )

        pot = cls(core, a_unit, field_mass, pos_ho, mass_ho, gram, G=G)
        bh = particles.black_holes
        for i in range(bh.n_particles):
            pot.add_black_hole(
                mass=float(bh.mass[i]), position=bh.pos[i], softening=bh_softening
            )
        if bh.n_particles:
            logger.info(f"Attached {bh.n_particles} black hole(s) to the potential")
        return pot

    # ---------------------------------------------------------- validation
    def validate(
        self,
        n_points: int = 2000,
        softening: Optional[float] = None,
        include_bh: bool = False,
        seed: int = 0,
    ) -> "ValidationResult":
        """Compare the disc-basis potential to direct summation.

        Probes a random subset of field-particle positions (appropriate for a
        flattened geometry) and reports the fractional agreement.

        Parameters
        ----------
        n_points : int, optional
            Number of probe points sampled from the field particles.
        softening : float, optional
            Softening for the direct sum, in HO units. Defaults to a mean
            interparticle-spacing estimate.
        include_bh : bool, optional
            If True the basis side includes the black-hole term. Left off by
            default so the check targets the field expansion.
        seed : int, optional
            Seed for the random probe-point selection.

        Returns
        -------
        result : ValidationResult
            The per-point and aggregate relative errors.
        """
        rng = np.random.default_rng(seed)
        n = len(self._field_pos_ho)
        idx = rng.choice(n, size=min(n_points, n), replace=False)
        pts = self._field_pos_ho[idx]

        if softening is None:
            r_hi = np.percentile(np.linalg.norm(self._field_pos_ho, axis=1), 90)
            softening = self._default_softening(r_hi)

        return self._validate_points(
            pts, np.linalg.norm(pts, axis=1), softening, include_bh
        )
