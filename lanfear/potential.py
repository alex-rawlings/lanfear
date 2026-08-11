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

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import _core
from .particle_system import ParticleSystem


@dataclass
class ValidationResult:
    """Summary of the SCF-vs-direct comparison (dimensionless relative errors)."""

    radii: np.ndarray  # HO-unit sample radii
    rel_error: np.ndarray  # |Phi_scf - Phi_direct| / |Phi_direct| per point
    median: float
    p90: float
    worst: float

    def passed(self, tolerance: float) -> bool:
        """True if the median relative error is below ``tolerance`` (fraction)."""
        return self.median < tolerance

    def __repr__(self) -> str:
        return (
            f"ValidationResult(median={self.median:.2%}, p90={self.p90:.2%}, "
            f"worst={self.worst:.2%}, n={len(self.rel_error)})"
        )


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
        """The underlying picklable C++ potential (for the orbit drivers)."""
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
        particles:
            A ParticleSystem that has already been :meth:`prepare`\\ d
            (recentred, aligned, scale radius estimated).
        n_max, l_max:
            Radial and angular truncation orders of the HO expansion.
        bh_softening:
            Plummer softening for each black hole, in units of the scale radius.
        """
        if particles.scale_radius is None:
            particles.estimate_scale_radius()
        a = particles.scale_radius

        field = particles.field
        field_mass = float(np.sum(field.mass))
        pos_ho = field.pos / a
        mass_ho = field.mass / field_mass

        scf = _core.SCFPotential(n_max, l_max, pos_ho, mass_ho)

        pot = cls(scf, a, field_mass, pos_ho, mass_ho, G=G)

        # Re-attach black holes at their true positions (HO units).
        bh = particles.black_holes
        for i in range(bh.n_particles):
            pot.add_black_hole(
                mass=float(bh.mass[i]),
                position=bh.pos[i],
                softening=bh_softening,
            )
        return pot

    def add_black_hole(self, mass, position, softening: float = 1e-3) -> None:
        """Add a softened point mass. ``mass``/``position`` are *physical*.

        ``softening`` is in scale-radius (HO) units.
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

    # ------------------------------------------------------------- units
    def to_ho_state(self, pos_phys: np.ndarray, vel_phys: np.ndarray) -> np.ndarray:
        """Convert physical positions/velocities to HO integration states.

        ``pos_phys`` (N,3) in length units, ``vel_phys`` (N,3) in velocity
        units (km/s for the default Gadget system). Returns an (N,6) array of
        ``(x, y, z, vx, vy, vz)`` in HO units, ready for ``integrate_batch``.
        """
        pos = np.atleast_2d(np.asarray(pos_phys, dtype=np.float64))
        vel = np.atleast_2d(np.asarray(vel_phys, dtype=np.float64))
        states = np.empty((len(pos), 6))
        states[:, :3] = pos / self.scale_radius
        states[:, 3:] = vel / self.velocity_unit
        return states

    def period_to_physical(self, period_ho: np.ndarray) -> np.ndarray:
        """Convert an HO-unit period/time to physical time units."""
        return np.asarray(period_ho) * self.time_unit

    # ---------------------------------------------------------- evaluation
    @property
    def n_black_holes(self) -> int:
        return self._scf.num_black_holes

    def potential(self, points: np.ndarray) -> np.ndarray:
        """Potential (HO units) at physical Cartesian points ``(N, 3)``."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64)) / self.scale_radius
        return self._scf.potential_batch(pts)

    def acceleration(self, points: np.ndarray) -> np.ndarray:
        """Acceleration (HO units) at physical Cartesian points ``(N, 3)``."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64)) / self.scale_radius
        return self._scf.acceleration_batch(pts)

    def _potential_ho(self, points_ho: np.ndarray) -> np.ndarray:
        return self._scf.potential_batch(np.ascontiguousarray(points_ho))

    # ---------------------------------------------------------- validation
    def _direct_potential_ho(
        self, points_ho: np.ndarray, softening: float
    ) -> np.ndarray:
        """Direct-summation potential of the field particles (HO units).

        This is the "actual" simulation potential the SCF fit is checked
        against. Evaluated in chunks to bound memory.
        """
        pos = self._field_pos_ho
        m = self._field_mass_ho
        eps2 = softening * softening
        out = np.empty(len(points_ho))
        chunk = max(1, int(2e7 // len(pos)))  # ~cap on the temporary (Npts x N)
        for lo in range(0, len(points_ho), chunk):
            hi = min(lo + chunk, len(points_ho))
            d = points_ho[lo:hi, None, :] - pos[None, :, :]
            r = np.sqrt(np.einsum("pij,pij->pi", d, d) + eps2)
            out[lo:hi] = -np.sum(m[None, :] / r, axis=1)
        return out

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
        r_range:
            (r_min, r_max) in *physical* units. Defaults to the 5th-95th
            percentile of the field-particle radii.
        softening:
            Softening for the direct sum, in HO units. Defaults to a mean
            interparticle spacing estimate, which tames discreteness noise.
        include_bh:
            If True the SCF side includes the black-hole term. Left off by
            default so the check targets the *field* expansion, which is what
            the SCF is responsible for representing.
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
        return ValidationResult(
            radii=radii,
            rel_error=rel_shell,
            median=float(np.median(rel)),
            p90=float(np.percentile(rel, 90)),
            worst=float(np.max(rel)),
        )

    def _bh_potential_ho(self, points_ho: np.ndarray) -> np.ndarray:
        """The black-hole-only potential (HO units), for isolating the field."""
        out = np.zeros(len(points_ho))
        for mass_ho, pos_ho, soft in self._bh_params:
            d = points_ho - pos_ho[None, :]
            r = np.sqrt(np.einsum("ij,ij->i", d, d) + soft * soft)
            out += -mass_ho / r
        return out
