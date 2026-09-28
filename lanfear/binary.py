"""Supermassive black-hole (SMBH) binary diagnostics.

A snapshot with two BH particles may hold a bound SMBH binary. How the binary is
represented in the analytical potential depends on how compact it is compared
with its sphere of influence:

* A binary with semimajor axis ``a`` smaller than its influence radius
  ``r_infl`` (a bound, and typically hard, binary) orbits on a timescale much
  shorter than any field orbit that stays outside it. To such orbits it acts
  like a single point mass ``M_1 + M_2`` at its centre of mass, which is the
  most robust *static* representation: freezing the two BHs at their snapshot
  positions instead imposes a two-centre field no star ever experiences, and
  makes the result depend on the orbital phase at the snapshot.
* A wide or unbound pair (``a >= r_infl``) is not yet a binary in that sense,
  and the two BHs are kept as separate softened point masses.

The choice is made by :func:`resolve_binary_treatment` and applied when a
potential attaches its black holes (the ``binary_treatment`` argument of
``from_particles``). The binary's semimajor axis is also what
:func:`lanfear.analyse_family` uses to flag *binary-interacting* orbits: those
whose pericentre reaches the binary (see
:attr:`lanfear.OrbitResults.binary_interacting`). No static potential can
represent such an orbit -- in reality it is scattered (slingshot) by the binary
-- so its classification is not trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._logging import get_logger
from .particle_system import ParticleSystem, _DEFAULT_G

logger = get_logger(__name__)

# Allowed values of the ``binary_treatment`` argument of the potential builders.
BINARY_TREATMENTS = ("auto", "point", "separate")


def check_binary_treatment(binary_treatment: str) -> None:
    """Raise if ``binary_treatment`` is not a recognised option.

    Parameters
    ----------
    binary_treatment : str
        The value to check (see :data:`BINARY_TREATMENTS`).

    Raises
    ------
    ValueError
        If ``binary_treatment`` is not one of :data:`BINARY_TREATMENTS`.
    """
    if binary_treatment not in BINARY_TREATMENTS:
        raise ValueError(
            f"binary_treatment must be one of {BINARY_TREATMENTS}, "
            f"got '{binary_treatment}'"
        )


def influence_radius(
    particles: ParticleSystem,
    bh_mass: float,
    centre=None,
    mass_factor: float = 2.0,
) -> float:
    """Radius enclosing ``mass_factor`` times the BH mass in field particles.

    The standard N-body definition of the sphere of influence (e.g. Merritt
    2013): the radius within which the enclosed field mass equals twice the
    black-hole (here, binary) mass. Unlike ``G M / sigma^2`` it needs no
    velocity-dispersion estimate. Every field (non-BH) species contributes.

    Parameters
    ----------
    particles : ParticleSystem
        The system; only its field particles are used.
    bh_mass : float
        Black-hole mass (the binary's total mass) in physical units.
    centre : array-like of float, optional
        (3,) centre to measure radii from. Defaults to the origin.
    mass_factor : float, optional
        Enclosed field mass, in units of ``bh_mass``, that defines the radius
        (default 2).

    Returns
    -------
    r_infl : float
        The influence radius in physical length units; ``inf`` if the field
        holds less than ``mass_factor * bh_mass`` in total.
    """
    field = particles.field
    r = field.radii(centre)
    order = np.argsort(r)
    cumulative = np.cumsum(field.mass[order])
    target = mass_factor * bh_mass
    if field.n_particles == 0 or cumulative[-1] < target:
        logger.warning(
            f"Field mass is below {mass_factor:g} x the BH mass; influence "
            f"radius is undefined (set to inf)"
        )
        return float("inf")
    idx = int(np.searchsorted(cumulative, target))
    return float(r[order][idx])


@dataclass
class BinaryProperties:
    """Orbital state of a two-BH system at the snapshot.

    Parameters
    ----------
    masses : numpy.ndarray
        (2,) BH masses in physical units.
    positions : numpy.ndarray
        (2, 3) BH positions in physical units.
    velocities : numpy.ndarray
        (2, 3) BH velocities in physical units.
    separation : float
        Current BH separation.
    semimajor_axis : float
        Semimajor axis of the relative (Keplerian) orbit, from the two-body
        energy; ``inf`` if the pair is unbound.
    eccentricity : float
        Eccentricity of the relative orbit (> 1 if unbound).
    influence_radius : float
        Radius enclosing twice the binary mass in field particles, measured
        from the binary's centre of mass (see :func:`influence_radius`).
    """

    masses: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    separation: float
    semimajor_axis: float
    eccentricity: float
    influence_radius: float

    @property
    def total_mass(self) -> float:
        """Combined BH mass ``M_1 + M_2``.

        Returns
        -------
        mass : float
            Total binary mass in physical units.
        """
        return float(np.sum(self.masses))

    @property
    def centre_of_mass(self) -> np.ndarray:
        """Mass-weighted BH position.

        Returns
        -------
        pos : numpy.ndarray
            (3,) centre of mass in physical units.
        """
        return np.average(self.positions, weights=self.masses, axis=0)

    @property
    def bound(self) -> bool:
        """Whether the two-body energy of the pair is negative.

        Returns
        -------
        bound : bool
            True if the semimajor axis is finite.
        """
        return bool(np.isfinite(self.semimajor_axis))

    @property
    def compact(self) -> bool:
        """Whether the pair is a bound binary inside its sphere of influence.

        This is the criterion ``binary_treatment="auto"`` uses to replace the
        two BHs by one point mass: ``a < r_infl``.

        Returns
        -------
        compact : bool
            True if bound and ``semimajor_axis < influence_radius``.
        """
        return self.bound and self.semimajor_axis < self.influence_radius


def binary_properties(
    particles: ParticleSystem,
    G: float = _DEFAULT_G,
    influence_mass_factor: float = 2.0,
) -> Optional[BinaryProperties]:
    """Orbital elements and influence radius of a two-BH snapshot.

    The relative orbit is treated as an isolated Keplerian two-body problem
    (the field's contribution to the pair's relative motion is neglected,
    which is accurate for a bound binary, whose enclosed field mass is small).

    Parameters
    ----------
    particles : ParticleSystem
        The system (physical units).
    G : float, optional
        Gravitational constant in the physical unit system (default Gadget).
    influence_mass_factor : float, optional
        Passed to :func:`influence_radius` as ``mass_factor``.

    Returns
    -------
    binary : BinaryProperties or None
        The binary's properties, or None unless the system holds exactly two
        BH particles.
    """
    bh = particles.black_holes
    if bh.n_particles != 2:
        return None

    masses = np.asarray(bh.mass, dtype=np.float64)
    mu = G * float(np.sum(masses))
    r_rel = bh.pos[0] - bh.pos[1]
    v_rel = bh.vel[0] - bh.vel[1]
    separation = float(np.linalg.norm(r_rel))
    speed_sq = float(np.dot(v_rel, v_rel))

    energy = 0.5 * speed_sq - mu / separation
    semimajor_axis = -mu / (2.0 * energy) if energy < 0.0 else float("inf")
    # Eccentricity vector e = (v x h) / mu - r / |r|, with h = r x v.
    h = np.cross(r_rel, v_rel)
    e_vec = np.cross(v_rel, h) / mu - r_rel / separation
    eccentricity = float(np.linalg.norm(e_vec))

    centre = np.average(bh.pos, weights=masses, axis=0)
    r_infl = influence_radius(
        particles,
        float(np.sum(masses)),
        centre=centre,
        mass_factor=influence_mass_factor,
    )
    return BinaryProperties(
        masses=masses,
        positions=np.array(bh.pos, dtype=np.float64),
        velocities=np.array(bh.vel, dtype=np.float64),
        separation=separation,
        semimajor_axis=float(semimajor_axis),
        eccentricity=eccentricity,
        influence_radius=r_infl,
    )


def resolve_binary_treatment(
    binary: Optional[BinaryProperties], binary_treatment: str
) -> bool:
    """Decide whether a two-BH system is attached as one combined point mass.

    Parameters
    ----------
    binary : BinaryProperties or None
        The binary (None unless the snapshot holds exactly two BHs).
    binary_treatment : {"auto", "point", "separate"}
        ``"auto"`` combines the pair only when it is a bound binary with
        ``a < r_infl`` (:attr:`BinaryProperties.compact`); ``"point"`` always
        combines it; ``"separate"`` never does.

    Returns
    -------
    combine : bool
        True if the two BHs should be attached as one point mass at their
        centre of mass.

    Raises
    ------
    ValueError
        If ``binary_treatment`` is not recognised.
    """
    check_binary_treatment(binary_treatment)
    if binary is None:
        return False
    if binary_treatment == "point":
        return True
    if binary_treatment == "separate":
        return False
    return binary.compact
