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

from typing import Optional

import numpy as np

from . import _core
from .particle_system import ParticleSystem
from .potential import ValidationResult


# NumPy Miyamoto-Nagai primitives, matching lanfear/disc_potential.hpp exactly
# (unit-tested against the C++ versions in tests/test_disc.py).
def _mn_potential(x, y, z, a, b):
    zeta = np.sqrt(z * z + b * b)
    return -1.0 / np.sqrt(x * x + y * y + (a + zeta) ** 2)


def _mn_density(R, z, a, b):
    zeta = np.sqrt(z * z + b * b)
    az = a + zeta
    num = b * b * (a * R * R + (a + 3.0 * zeta) * az * az)
    den = 4.0 * np.pi * (R * R + az * az) ** 2.5 * zeta**3
    return num / den


class DiscPotential:
    """Analytical disc potential (MN basis field + softened black holes)."""

    DEFAULT_G = 43009.1  # Gadget units (kpc, 1e10 Msun, km/s)

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
        self._disc = core
        self.scale_radius = scale_radius
        self.field_mass = field_mass
        self.G = G
        self.velocity_unit = np.sqrt(G * field_mass / scale_radius)
        self.time_unit = scale_radius / self.velocity_unit
        self.gram = gram
        self._field_pos_ho = field_pos_ho
        self._field_mass_ho = field_mass_ho
        self._bh_params: list = []

    @property
    def core(self):
        """The underlying picklable C++ potential (for the orbit drivers)."""
        return self._disc

    # ------------------------------------------------------------- basis
    @staticmethod
    def default_basis(pos_ho, n_radial=8, n_vert=3):
        """A geometric grid of MN (radial, thickness) scales spanning the disc.

        ``pos_ho`` are field positions in HO units. Radial scales span roughly
        the cylindrical extent; thickness scales span the vertical extent.
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
    def _gram_matrix(a_list, b_list, n_quad=500):
        """Gram matrix gram[i,j] = integral rho_i Phi_j dV (meridional quadrature).

        Uses log-spaced nodes in R and z (midpoint rule in ln R, ln z) so that
        every basis scale -- which span more than two decades -- is resolved with
        roughly equal accuracy. The MN density is z-symmetric, so z runs over the
        positive half and the result is doubled. Contributions from R,z below the
        inner cutoff are negligible because dV ~ R -> 0 there.
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
    def _solve(gram, b, rcond):
        """Solve gram @ c = b via a truncated pseudo-inverse.

        The MN basis is highly collinear, so the Gram matrix is severely
        ill-conditioned. -gram is symmetric (positive definite in exact
        arithmetic), so we eigendecompose it and drop modes whose eigenvalue is
        below ``rcond * lambda_max`` -- keeping only the well-determined
        combinations of basis functions. This yields a stable, smooth fit.
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

        ``rcond`` sets the eigenvalue-truncation level of the (ill-conditioned)
        Gram solve: smaller keeps more basis modes (better fit, less stable),
        larger is smoother/more robust.
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

        core = _core.DiscPotential(a_arr, b_arr)
        b = np.asarray(core.scf_sum(pos_ho, mass_ho))
        gram = cls._gram_matrix(a_arr, b_arr)
        c = cls._solve(gram, b, rcond)
        core.set_coefficients(np.ascontiguousarray(c))

        pot = cls(core, a_unit, field_mass, pos_ho, mass_ho, gram, G=G)
        bh = particles.black_holes
        for i in range(bh.n_particles):
            pot.add_black_hole(
                mass=float(bh.mass[i]), position=bh.pos[i], softening=bh_softening
            )
        return pot

    def add_black_hole(self, mass, position, softening: float = 1e-3) -> None:
        pos_ho = np.asarray(position, dtype=np.float64) / self.scale_radius
        mass_ho = mass / self.field_mass
        self._disc.add_black_hole(
            mass_ho, float(pos_ho[0]), float(pos_ho[1]), float(pos_ho[2]), softening
        )
        self._bh_params.append((mass_ho, pos_ho, softening))

    @property
    def n_black_holes(self) -> int:
        return self._disc.num_black_holes

    # ---------------------------------------------------------- evaluation
    def to_ho_state(self, pos_phys, vel_phys):
        pos = np.atleast_2d(np.asarray(pos_phys, dtype=np.float64))
        vel = np.atleast_2d(np.asarray(vel_phys, dtype=np.float64))
        states = np.empty((len(pos), 6))
        states[:, :3] = pos / self.scale_radius
        states[:, 3:] = vel / self.velocity_unit
        return states

    def potential(self, points):
        pts = np.atleast_2d(np.asarray(points, float)) / self.scale_radius
        return self._disc.potential_batch(pts)

    def acceleration(self, points):
        pts = np.atleast_2d(np.asarray(points, float)) / self.scale_radius
        return self._disc.acceleration_batch(pts)

    # ---------------------------------------------------------- validation
    def _bh_potential_ho(self, points_ho):
        out = np.zeros(len(points_ho))
        for mass_ho, pos_ho, soft in self._bh_params:
            d = points_ho - pos_ho[None, :]
            out += -mass_ho / np.sqrt(np.einsum("ij,ij->i", d, d) + soft * soft)
        return out

    def _direct_potential_ho(self, points_ho, softening):
        pos = self._field_pos_ho
        m = self._field_mass_ho
        eps2 = softening * softening
        out = np.empty(len(points_ho))
        chunk = max(1, int(2e7 // len(pos)))
        for lo in range(0, len(points_ho), chunk):
            hi = min(lo + chunk, len(points_ho))
            d = points_ho[lo:hi, None, :] - pos[None, :, :]
            r = np.sqrt(np.einsum("pij,pij->pi", d, d) + eps2)
            out[lo:hi] = -np.sum(m[None, :] / r, axis=1)
        return out

    def validate(
        self,
        n_points: int = 2000,
        softening: Optional[float] = None,
        include_bh: bool = False,
        seed: int = 0,
    ) -> ValidationResult:
        """Compare the disc-basis potential to direct summation.

        Probes a random subset of field-particle positions (appropriate for a
        flattened geometry) and reports the fractional agreement.
        """
        rng = np.random.default_rng(seed)
        n = len(self._field_pos_ho)
        idx = rng.choice(n, size=min(n_points, n), replace=False)
        pts = self._field_pos_ho[idx]

        if softening is None:
            r_hi = np.percentile(np.linalg.norm(self._field_pos_ho, axis=1), 90)
            n_in = max(1, np.sum(np.linalg.norm(self._field_pos_ho, axis=1) < r_hi))
            softening = float(r_hi / n_in ** (1.0 / 3.0))

        phi_basis = self._disc.potential_batch(np.ascontiguousarray(pts))
        if not include_bh and self.n_black_holes > 0:
            phi_basis = phi_basis - self._bh_potential_ho(pts)
        phi_direct = self._direct_potential_ho(pts, softening)

        rel = np.abs(phi_basis - phi_direct) / np.abs(phi_direct)
        return ValidationResult(
            radii=np.linalg.norm(pts, axis=1),
            rel_error=rel,
            median=float(np.median(rel)),
            p90=float(np.percentile(rel, 90)),
            worst=float(np.max(rel)),
        )
