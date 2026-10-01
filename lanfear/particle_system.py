"""Particle data container and Gadget-4 HDF5 reader.

A :class:`ParticleSystem` holds positions, velocities, masses, IDs and a
per-particle species label. It knows how to load a Gadget-4 HDF5 snapshot,
recentre and align the system, estimate the pattern speed of a tumbling figure,
and hand off the field (non-BH) particles to the SCF potential in
Hernquist-Ostriker (HO) units.

HO units: ``G = M_field = scale_radius = 1``. The scale radius is estimated from
the field half-mass radius as ``r_half / (1 + sqrt(2))`` (exact for a Hernquist
profile), and the mass unit is the total field (non-BH) mass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from . import _core
from ._logging import get_logger

logger = get_logger(__name__)

# Gravitational constant in the default Gadget unit system (kpc, 1e10 Msun,
# km/s); matches Potential.DEFAULT_G in potential.py.
_DEFAULT_G = 43009.1

# Gadget PartType -> species label used throughout the code.
_PARTTYPE_TO_SPECIES = {
    "PartType0": "GAS",
    "PartType1": "DM",
    "PartType2": "DISK",
    "PartType3": "BULGE",
    "PartType4": "STAR",
    "PartType5": "BH",
}


def _as_pattern_speed(value) -> np.ndarray:
    """Normalise a user-supplied pattern speed to an angular-velocity vector.

    Parameters
    ----------
    value : None, str, float or array-like of float
        ``None`` or ``"none"`` (any case) for no figure rotation; a scalar for
        rotation about the z (short, after :meth:`ParticleSystem.align`) axis;
        or a (3,) angular-velocity vector. Physical units (velocity unit /
        length unit, e.g. km/s/kpc).

    Returns
    -------
    omega : numpy.ndarray
        (3,) angular-velocity vector (all zero for no rotation).

    Raises
    ------
    ValueError
        If ``value`` is an unrecognised string, has the wrong shape, or is not
        finite.
    """
    if value is None:
        return np.zeros(3)
    if isinstance(value, str):
        if value.lower() == "none":
            return np.zeros(3)
        raise ValueError(
            f"pattern_speed must be 'estimate', 'none', a number or a (3,) "
            f"vector, got {value!r}"
        )
    omega = np.asarray(value, dtype=np.float64)
    if omega.ndim == 0:
        omega = np.array([0.0, 0.0, float(omega)])
    if omega.shape != (3,) or not np.all(np.isfinite(omega)):
        raise ValueError(
            f"pattern_speed must be a finite number or (3,) vector, got {value!r}"
        )
    return omega.copy()


def _gaussian_window(r: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth radial window ``g(r) = exp(-r^2 / (2 R^2))`` and its slope.

    Depends on position only, and falls smoothly to zero rather than cutting
    particles off at a hard edge (see :func:`_windowed_pattern_speed`).

    Parameters
    ----------
    r : numpy.ndarray
        (N,) radii from the origin.
    radius : float
        Window scale ``R``, in the same units as ``r``.

    Returns
    -------
    window : numpy.ndarray
        (N,) ``g(r)``.
    slope : numpy.ndarray
        (N,) ``dg/dr``.
    """
    s = r / radius
    window = np.exp(-0.5 * s**2)
    return window, -s / radius * window


def _windowed_principal_axes(
    unit: np.ndarray, weight: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Principal axes of the weighted shape tensor ``sum weight u u^T``.

    Parameters
    ----------
    unit : numpy.ndarray
        (N, 3) unit position vectors ``u = x / r``.
    weight : numpy.ndarray
        (N,) weights (``m r g(r)`` for the windowed tensor).

    Returns
    -------
    eigenvalues : numpy.ndarray
        (3,) of the tensor normalised to unit trace, descending.
    axes : numpy.ndarray
        (3, 3) principal axes as columns (long, intermediate, short),
        forming a proper rotation so that angular velocities stay vectors.
    """
    tensor = np.einsum("k,ki,kj->ij", weight / weight.sum(), unit, unit)
    vals, vecs = np.linalg.eigh(tensor)
    order = np.argsort(vals)[::-1]  # long, intermediate, short
    vals, vecs = vals[order], vecs[:, order]
    if np.linalg.det(vecs) < 0:
        vecs[:, 2] *= -1
    return vals, vecs


def _windowed_pattern_speed(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    window: np.ndarray,
    window_slope: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Pattern speed from the rate of change of a radially windowed tensor.

    The tensor is ``T = sum m r g(r) u u^T`` (``u = x / r``) with the given
    window ``g``. For a steady figure rotating rigidly at ``Omega``, splitting
    each velocity into ``Omega x x`` plus the rotating-frame velocity ``v'``
    gives ``dT/dt = [Omega x, T] + sum m v' . grad(r g u u^T)``. The second term
    is minus the window-weighted divergence of the steady rotating-frame mass
    flux, and vanishes on average. That needs two things: no particle may be
    selected by its velocity (or by any quantity not conserved in the rotating
    frame, such as the inertial energy), and the window must fall smoothly to
    zero rather than cut particles off at a hard edge, where a flux would cross.

    Parameters
    ----------
    pos, vel : numpy.ndarray
        (N, 3) positions and velocities about the (recentred) origin.
    mass : numpy.ndarray
        (N,) particle masses.
    window : numpy.ndarray
        (N,) smooth window ``g(r)`` at each particle, depending on position only.
    window_slope : numpy.ndarray
        (N,) its radial derivative ``dg/dr`` at each particle.

    Returns
    -------
    omega : numpy.ndarray
        (3,) angular velocity about each principal axis (NaN where undefined).
    sigma : numpy.ndarray
        (3,) Poisson uncertainty of each component.
    eigenvalues : numpy.ndarray
        (3,) of the normalised tensor, descending.
    axes : numpy.ndarray
        (3, 3) principal axes as columns (long, intermediate, short), forming
        a proper rotation.
    n : int
        Number of particles with non-negligible weight.
    """
    r = np.linalg.norm(pos, axis=1)
    # Particles far out in the window's tail carry no weight; dropping them
    # only saves work (the window is already ~0 there, so no edge is created).
    keep = (r > 0) & (window > 1e-10)
    n = int(np.sum(keep))
    nan3 = np.full(3, np.nan)
    if n < 10:
        return nan3, nan3.copy(), nan3.copy(), np.eye(3), n
    p, v, m, r = pos[keep], vel[keep], mass[keep], r[keep]
    g, dg_dr = window[keep], window_slope[keep]
    weight = m * r * g
    weight_sum = weight.sum()  # normalise so the eigenvalues sum to 1
    u = p / r[:, None]

    vals, vecs = _windowed_principal_axes(u, weight)

    # Unit vectors, radial and transverse velocities in the principal frame.
    u_p = u @ vecs
    v_p = v @ vecs
    v_radial = np.sum(u_p * v_p, axis=1)
    v_transverse = v_p - u_p * v_radial[:, None]
    radial_factor = m * (g + r * dg_dr) * v_radial / weight_sum
    transverse_factor = m * g / weight_sum

    omega = nan3.copy()
    sigma = nan3.copy()
    for i, j, k in ((1, 2, 0), (2, 0, 1), (0, 1, 2)):
        contribution = radial_factor * u_p[:, i] * u_p[:, j] + transverse_factor * (
            v_transverse[:, i] * u_p[:, j] + u_p[:, i] * v_transverse[:, j]
        )
        rate = contribution.sum()
        rate_error = np.sqrt(n * np.var(contribution, ddof=1))
        gap = vals[i] - vals[j]
        if gap != 0.0:
            omega[k] = rate / gap
            sigma[k] = rate_error / abs(gap)
    return omega, sigma, vals, vecs, n


@dataclass
class ParticleSystem:
    """A collection of simulation particles.

    All arrays are indexed consistently: ``pos``/``vel`` have shape ``(N, 3)``
    and ``mass``/``ids``/``species`` have shape ``(N,)``.

    Parameters
    ----------
    pos : numpy.ndarray
        (N, 3) particle positions.
    vel : numpy.ndarray
        (N, 3) particle velocities.
    mass : numpy.ndarray
        (N,) particle masses.
    ids : numpy.ndarray
        (N,) particle IDs.
    species : numpy.ndarray
        (N,) species labels (e.g. ``"STAR"``, ``"DM"``, ``"BH"``).
    scale_radius : float, optional
        Cached HO scale radius; populated by :meth:`estimate_scale_radius`.
    source_file : str, optional
        Path the system was loaded from; populated by :meth:`from_gadget_hdf5`.
    centring : str, optional
        The ``on``/``centre`` selector last used by :meth:`recentre`; ``None``
        if the system has not been recentred.
    pattern_speed : numpy.ndarray, optional
        (3,) angular velocity of the figure (its pattern speed), in physical
        units (velocity unit / length unit, e.g. km/s/kpc) and in the current
        coordinate frame. Set by :meth:`prepare` (estimated by default, see
        :meth:`estimate_pattern_speed`) and inherited by the potentials built
        from this system, which then rotate rigidly at this rate. ``None``
        (not set) and all-zero both mean a static figure.
    """

    pos: np.ndarray
    vel: np.ndarray
    mass: np.ndarray
    ids: np.ndarray
    species: np.ndarray
    scale_radius: Optional[float] = field(default=None)
    source_file: Optional[str] = field(default=None)
    centring: Optional[str] = field(default=None)
    pattern_speed: Optional[np.ndarray] = field(default=None)

    # ------------------------------------------------------------------ IO
    @classmethod
    def from_gadget_hdf5(cls, filename: str) -> "ParticleSystem":
        """Load a Gadget-4 HDF5 snapshot.

        Reads every ``PartTypeX`` group present, taking ``Coordinates``,
        ``Velocities``, ``ParticleIDs`` and masses (per-particle ``Masses`` if
        present, otherwise the constant from the header ``MassTable``).

        Parameters
        ----------
        filename : str
            Path to the Gadget-4 HDF5 snapshot file.

        Returns
        -------
        system : ParticleSystem
            All particle groups concatenated into a single system.

        Raises
        ------
        ValueError
            If the file contains no particle group with ``Coordinates``.
        """
        import h5py

        pos, vel, mass, ids, species = [], [], [], [], []
        with h5py.File(filename, "r") as f:
            mass_table = None
            if "Header" in f:
                mass_table = f["Header"].attrs.get("MassTable", None)
            for key in f.keys():
                if not key.startswith("PartType"):
                    continue
                grp = f[key]
                if "Coordinates" not in grp:
                    continue
                n = grp["Coordinates"].shape[0]
                pos.append(np.asarray(grp["Coordinates"][:], dtype=np.float64))
                vel.append(np.asarray(grp["Velocities"][:], dtype=np.float64))
                if "Masses" in grp:
                    m = np.asarray(grp["Masses"][:], dtype=np.float64)
                else:
                    ptype = int(key.replace("PartType", ""))
                    const = mass_table[ptype] if mass_table is not None else 0.0
                    m = np.full(n, const, dtype=np.float64)
                mass.append(m)
                if "ParticleIDs" in grp:
                    ids.append(np.asarray(grp["ParticleIDs"][:]))
                else:
                    ids.append(np.arange(n, dtype=np.int64))
                species.append(np.full(n, _PARTTYPE_TO_SPECIES.get(key, key)))

        if not pos:
            raise ValueError(f"No particle groups with Coordinates in {filename}")

        system = cls(
            pos=np.concatenate(pos),
            vel=np.concatenate(vel),
            mass=np.concatenate(mass),
            ids=np.concatenate(ids),
            species=np.concatenate(species),
            source_file=str(filename),
        )
        logger.info(f"Loaded {system.n_particles} particles from {filename}")
        labels, counts = np.unique(system.species, return_counts=True)
        breakdown = {str(s): int(c) for s, c in zip(labels, counts)}
        logger.debug(f"Species breakdown: {breakdown}")
        return system

    # -------------------------------------------------------------- slicing
    @property
    def n_particles(self) -> int:
        """Number of particles in the system.

        Returns
        -------
        n : int
            Total particle count.
        """
        return len(self.mass)

    def select(self, mask) -> "ParticleSystem":
        """Return a new ParticleSystem containing the masked particles.

        Parameters
        ----------
        mask : numpy.ndarray or slice
            Boolean mask, integer index array, or slice selecting particles.

        Returns
        -------
        system : ParticleSystem
            A new system holding only the selected particles (carrying over the
            current scale radius, source file, centring and pattern speed).
        """
        mask = np.asarray(mask)
        return ParticleSystem(
            pos=self.pos[mask],
            vel=self.vel[mask],
            mass=self.mass[mask],
            ids=self.ids[mask],
            species=self.species[mask],
            scale_radius=self.scale_radius,
            source_file=self.source_file,
            centring=self.centring,
            pattern_speed=self.pattern_speed,
        )

    def random_subset(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
    ) -> "ParticleSystem":
        """Return a new ParticleSystem with ``n`` randomly-chosen particles.

        Particles are drawn without replacement and the original ordering is
        preserved. If ``n`` is at least the particle count the whole system is
        returned.

        Parameters
        ----------
        n : int
            Number of particles to keep.
        rng : numpy.random.Generator, optional
            Random generator to draw with. Defaults to a fresh
            ``numpy.random.default_rng()`` (unseeded) if not provided.

        Returns
        -------
        system : ParticleSystem
            A new system holding the sampled particles (carrying over the
            current scale radius).

        Raises
        ------
        ValueError
            If ``n`` is negative.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if rng is None:
            rng = np.random.default_rng()
        if n >= self.n_particles:
            return self.select(np.arange(self.n_particles))
        idx = np.sort(rng.choice(self.n_particles, size=int(n), replace=False))
        return self.select(idx)

    def species_mask(self, *labels: str) -> np.ndarray:
        """Boolean mask selecting the given species labels.

        Parameters
        ----------
        *labels : str
            One or more species labels (e.g. ``"STAR"``, ``"DM"``, ``"BH"``).

        Returns
        -------
        mask : numpy.ndarray
            (N,) boolean array, True where the particle species is in ``labels``.
        """
        want = set(labels)
        return np.array([s in want for s in self.species])

    def radius_mask(
        self,
        r_max: float,
        r_min: float = 0.0,
        centre=None,
    ) -> np.ndarray:
        """Boolean mask selecting particles in a radial range.

        Designed to compose with :meth:`select` (like :meth:`species_mask`), so
        integration and classification can be restricted to a spatial region
        while the potential is still built from the *whole* system. For example,
        to integrate only the stars inside ``r``::

            pot = lf.Potential.from_particles(ps)          # all particles
            inner = ps.select(ps.radius_mask(r))           # subset within r
            res = lf.analyse_family(pot, inner, family="STAR")

        Masks combine with the usual boolean operators, e.g.
        ``ps.select(ps.species_mask("STAR") & ps.radius_mask(r))``.

        Parameters
        ----------
        r_max : float
            Upper radius; particles with ``|r - centre| < r_max`` are selected.
        r_min : float, optional
            Lower radius for a shell selection (default 0). Particles with
            ``|r - centre| >= r_min`` are selected.
        centre : array-like of float, optional
            The (3,) reference point. Defaults to the origin (the centre after
            :meth:`prepare`/:meth:`recentre`).

        Returns
        -------
        mask : numpy.ndarray
            (N,) boolean array, True where ``r_min <= |r - centre| < r_max``.
        """
        r = self.radii(centre)
        return (r >= r_min) & (r < r_max)

    @property
    def field(self) -> "ParticleSystem":
        """The field particles: everything except black holes.

        Returns
        -------
        system : ParticleSystem
            A new system containing all non-BH particles.
        """
        return self.select(self.species != "BH")

    @property
    def black_holes(self) -> "ParticleSystem":
        """The black-hole particles.

        Returns
        -------
        system : ParticleSystem
            A new system containing only the BH particles.
        """
        return self.select(self.species == "BH")

    # ----------------------------------------------------------- geometry
    def radii(self, centre=None) -> np.ndarray:
        """Distance of each particle from a reference point.

        Parameters
        ----------
        centre : array-like of float, optional
            The (3,) reference point. Defaults to the origin.

        Returns
        -------
        r : numpy.ndarray
            (N,) radial distances.
        """
        centre = np.zeros(3) if centre is None else np.asarray(centre)
        return np.linalg.norm(self.pos - centre, axis=1)

    def centre_of_mass(
        self, species: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Mass-weighted centre of mass and mean velocity.

        Parameters
        ----------
        species : str, optional
            If given, restrict the average to this species label; otherwise use
            all particles.

        Returns
        -------
        pos_com : numpy.ndarray
            (3,) mass-weighted mean position.
        vel_com : numpy.ndarray
            (3,) mass-weighted mean velocity.
        """
        if species is None:
            p, m = self.pos, self.mass
        else:
            mask = self.species == species
            p, m = self.pos[mask], self.mass[mask]
        pos_com = np.average(p, weights=m, axis=0)
        vel_com = np.average(
            self.vel if species is None else self.vel[self.species == species],
            weights=m,
            axis=0,
        )
        return pos_com, vel_com

    def half_mass_radius(self) -> float:
        """Spherical half-(field-)mass radius about the current origin.

        Returns
        -------
        r_half : float
            Radius enclosing half of the total field (non-BH) mass.
        """
        fld = self.field
        r = fld.radii()
        order = np.argsort(r)
        cumulative = np.cumsum(fld.mass[order])
        half = cumulative[-1] / 2.0
        idx = np.searchsorted(cumulative, half)
        return float(r[order][min(idx, len(r) - 1)])

    def estimate_scale_radius(self) -> float:
        """Estimate and cache the Hernquist scale radius.

        Uses ``r_half / (1 + sqrt(2))`` (exact for a Hernquist profile) and
        stores the result in :attr:`scale_radius`.

        Returns
        -------
        scale_radius : float
            The estimated scale radius.
        """
        self.scale_radius = self.half_mass_radius() / (1.0 + np.sqrt(2.0))
        logger.info(f"Estimated scale radius: {self.scale_radius:.4g}")
        return self.scale_radius

    # --------------------------------------------------------- preparation
    def shrinking_sphere_centre(
        self,
        enclose_frac: float = 0.80,
        shrink_factor: float = 0.93,
        stop_frac: float = 0.01,
        use: str = "field",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Locate the centre by the shrinking-sphere method (C++ core).

        Starting from the naive mass-weighted centre of mass and a sphere
        enclosing ``enclose_frac`` of the particles, the centre is iteratively
        recomputed as the mass-weighted COM of the particles inside the sphere
        while the sphere radius shrinks by ``shrink_factor`` each step, until the
        sphere holds no more than ``stop_frac`` of the particles. This is robust
        to substructure and asymmetric outskirts that bias a single global COM
        (Power et al. 2003). The velocity centre is the mass-weighted mean
        velocity of the particles in the final sphere.

        Parameters
        ----------
        enclose_frac : float, optional
            Fraction of particles enclosed by the initial sphere (default 0.80).
        shrink_factor : float, optional
            Factor the sphere radius is multiplied by each step (default 0.93).
        stop_frac : float, optional
            Iteration stops once the sphere holds no more than this fraction of
            the particles (default 0.01); at least one particle is required.
        use : str, optional
            Which subset defines the centre: ``"field"`` (non-BH particles, the
            default), ``"all"``, or a species label such as ``"STAR"``.

        Returns
        -------
        pos_centre : numpy.ndarray
            (3,) refined spatial centre.
        vel_centre : numpy.ndarray
            (3,) bulk velocity (COM velocity of the final sphere).

        Raises
        ------
        ValueError
            If no particles match the ``use`` selection.
        """
        if use == "field":
            mask = self.species != "BH"
        elif use == "all":
            mask = np.ones(self.n_particles, dtype=bool)
        else:
            mask = self.species == use
        if not np.any(mask):
            raise ValueError(f"No particles matched centre selection '{use}'")
        result = _core.shrinking_sphere_centre(
            np.ascontiguousarray(self.pos[mask], dtype=np.float64),
            np.ascontiguousarray(self.vel[mask], dtype=np.float64),
            np.ascontiguousarray(self.mass[mask], dtype=np.float64),
            enclose_frac,
            shrink_factor,
            stop_frac,
        )
        pos_centre = np.asarray(result["position"])
        vel_centre = np.asarray(result["velocity"])
        logger.debug(
            f"Shrinking-sphere centre ({use}): pos={np.round(pos_centre, 4)} "
            f"vel={np.round(vel_centre, 4)} ({result['n_iterations']} iterations, "
            f"{result['n_final']} particles in final sphere of radius "
            f"{result['radius']:.4g})"
        )
        return pos_centre, vel_centre

    def recentre(self, on: str = "bh") -> None:
        """Shift positions/velocities so the chosen centre is the origin.

        Modifies the system in place.

        Parameters
        ----------
        on : str, optional
            How to define the centre: ``"shrinking_sphere"`` (the shrinking-
            sphere centre of the field particles; see
            :meth:`shrinking_sphere_centre`), ``"field"`` (field COM), ``"bh"``
            (black-hole COM, the default), or a species label such as
            ``"STAR"``. Systems with no BH particles must pass an explicit
            alternative (e.g. ``"shrinking_sphere"``) rather than relying on
            the default.

        Raises
        ------
        ValueError
            If no particles match the requested centre selection.
        """
        if on == "shrinking_sphere":
            pos_com, vel_com = self.shrinking_sphere_centre()
        else:
            if on == "field":
                mask = self.species != "BH"
            elif on == "bh":
                mask = self.species == "BH"
            else:
                mask = self.species == on
            if not np.any(mask):
                raise ValueError(f"No particles matched centre selection '{on}'")
            pos_com = np.average(self.pos[mask], weights=self.mass[mask], axis=0)
            vel_com = np.average(self.vel[mask], weights=self.mass[mask], axis=0)
        self.pos = self.pos - pos_com
        self.vel = self.vel - vel_com
        self.centring = on
        logger.debug(
            f"Recentred on '{on}'; shifted position COM by {np.round(pos_com, 4)}"
        )

    def align(self, window_radius: Optional[float] = None) -> np.ndarray:
        """Rotate so the field principal axes align with x, y, z (in place).

        Diagonalises the shape tensor ``T = sum m r g(r) u u^T`` of every field
        particle (``u = x / r``), weighted by the smooth radial window
        ``g(r) = exp(-r^2 / (2 R^2))`` with ``R = window_radius``: the longest
        axis maps to x and the shortest to z. This is the tensor that
        :meth:`estimate_pattern_speed` uses (with the same default window), so
        after alignment its principal axes are the x, y, z axes.

        The window keeps the diffuse, often asymmetric outskirts and tidal
        debris from biasing the shape. It depends on position only: no particle
        is selected by its velocity (e.g. by energy), which would make the axes
        depend on the kinematics and, in a tumbling figure, on the sense of
        rotation. Assumes the system has already been recentred.

        Parameters
        ----------
        window_radius : float, optional
            Scale ``R`` of the radial window (physical length units). Defaults
            to the field half-mass radius about the origin.

        Returns
        -------
        rotation : numpy.ndarray
            The (3, 3) rotation matrix applied to positions and velocities
            (and to :attr:`pattern_speed`, if already set).
        """
        fld = self.field
        if window_radius is None:
            window_radius = self.half_mass_radius()
        r = np.linalg.norm(fld.pos, axis=1)
        good = r > 0
        window, _ = _gaussian_window(r[good], float(window_radius))
        _, axes = _windowed_principal_axes(
            fld.pos[good] / r[good, None], fld.mass[good] * r[good] * window
        )
        rot = axes.T  # rows: long, intermediate, short axes -> x, y, z
        self.pos = self.pos @ rot.T
        self.vel = self.vel @ rot.T
        if self.pattern_speed is not None:
            self.pattern_speed = rot @ self.pattern_speed
        logger.debug(
            f"Aligned field principal axes with x, y, z using {fld.n_particles} "
            f"field particles (window radius {float(window_radius):.4g})"
        )
        return rot

    def detect_figure_rotation(
        self,
        axis_ratio_threshold: float = 0.9,
        rotation_threshold: float = 0.3,
    ) -> dict:
        """Heuristically flag likely figure rotation (a tumbling figure).

        Figure rotation is the pattern speed at which a non-axisymmetric
        figure tumbles. :meth:`estimate_pattern_speed` measures it, and
        :meth:`prepare` uses that measurement by default. This method is the
        cheaper kinematic heuristic that :meth:`prepare` falls back on when
        orbits are to be integrated in a *static* potential (no pattern speed
        set). It flags the regime in which figure rotation is likely and a
        static potential would therefore mis-assign families: a
        non-axisymmetric field (in-plane axis ratio ``b/a`` below
        ``axis_ratio_threshold``) that also shows significant ordered rotation
        about its short axis (``|v_rot| / sigma`` above ``rotation_threshold``).
        Non-rotating triaxial systems are box/tube dominated and carry little net
        rotation, so strong rotation in a triaxial figure is the tell-tale sign.

        The field principal axes and rotation are computed here from the
        mass-weighted shape tensor, so the result does not depend on the system
        having been aligned first. A warning is logged when rotation is detected.

        Parameters
        ----------
        axis_ratio_threshold : float, optional
            The field counts as non-axisymmetric when the intermediate-to-long
            axis ratio ``b/a`` is below this value.
        rotation_threshold : float, optional
            Ordered rotation counts as significant when ``|v_rot| / sigma``
            (mean rotation speed about the short axis over the 1-D velocity
            dispersion) exceeds this value.

        Returns
        -------
        result : dict
            Diagnostics with keys ``"detected"`` (bool), ``"b_over_a"`` and
            ``"c_over_a"`` (float axis ratios), ``"rotation_measure"`` (float,
            ``|v_rot| / sigma``), ``"short_axis"`` (numpy.ndarray, the short
            principal axis), and ``"L_short_fraction"`` (float, fraction of the
            angular momentum aligned with the short axis).
        """
        fld = self.field
        nan_result = {
            "detected": False,
            "b_over_a": float("nan"),
            "c_over_a": float("nan"),
            "rotation_measure": float("nan"),
            "short_axis": np.array([0.0, 0.0, 1.0]),
            "L_short_fraction": float("nan"),
        }
        if fld.n_particles < 100:
            logger.debug("Too few field particles to assess figure rotation")
            return nan_result

        # Robust centre: the median, refined by the inner-90% mass mean. This
        # prevents a few extreme-radius outliers (which can dominate a plain
        # mass-weighted mean) from offsetting the centre and faking anisotropy.
        centre = np.median(fld.pos, axis=0)
        r = np.linalg.norm(fld.pos - centre, axis=1)
        sel = r <= np.percentile(r, 90)
        centre = np.average(fld.pos[sel], weights=fld.mass[sel], axis=0)
        r = np.linalg.norm(fld.pos - centre, axis=1)
        sel = r <= np.percentile(r, 90)

        # All subsequent quantities use the inlier aperture, which also excludes
        # the extended tail whose <r^2> would otherwise dominate the shape.
        m = fld.mass[sel]
        m_tot = m.sum()
        pos = fld.pos[sel] - centre
        vel = fld.vel[sel] - np.average(fld.vel[sel], weights=m, axis=0)

        # Shape tensor -> principal axes (descending: long, int, short).
        shape = np.einsum("k,ki,kj->ij", m, pos, pos) / m_tot
        vals, vecs = np.linalg.eigh(shape)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        a2, b2, c2 = np.maximum(vals, 0.0)
        b_over_a = float(np.sqrt(b2 / max(a2, 1e-300)))
        c_over_a = float(np.sqrt(c2 / max(a2, 1e-300)))
        short = vecs[:, 2]

        # Angular momentum and its alignment with the short axis.
        ell_vec = np.cross(pos, vel)  # specific angular momentum per particle
        L = np.einsum("k,ki->i", m, ell_vec)
        L_mag = np.linalg.norm(L)
        L_short_frac = float(abs(np.dot(L, short)) / L_mag) if L_mag > 0 else 0.0

        # Ordered rotation about the short axis vs velocity dispersion.
        ell_short = ell_vec @ short  # R * v_phi per particle
        r_perp = pos - np.outer(pos @ short, short)
        R = np.linalg.norm(r_perp, axis=1)
        good = R > 0
        denom = np.sum(m[good] * R[good])
        v_rot = float(np.sum(m[good] * ell_short[good]) / denom) if denom > 0 else 0.0
        sigma = float(np.sqrt(np.sum(m * np.sum(vel**2, axis=1)) / (3.0 * m_tot)))
        rotation_measure = abs(v_rot) / sigma if sigma > 0 else 0.0

        non_axisymmetric = b_over_a < axis_ratio_threshold
        detected = bool(non_axisymmetric and rotation_measure > rotation_threshold)
        result = {
            "detected": detected,
            "b_over_a": b_over_a,
            "c_over_a": c_over_a,
            "rotation_measure": rotation_measure,
            "short_axis": short,
            "L_short_fraction": L_short_frac,
        }
        if detected:
            logger.warning(
                f"Possible figure rotation: non-axisymmetric field "
                f"(b/a={b_over_a:.2f}) with significant ordered rotation about "
                f"its short axis (v_rot/sigma={rotation_measure:.2f}). With no "
                f"pattern speed set, orbits are integrated in a STATIC potential "
                f"and families will be mis-assigned if the figure is tumbling; "
                f"estimate the pattern speed (prepare(pattern_speed='estimate'), "
                f"see estimate_pattern_speed()) or set it explicitly."
            )
        else:
            logger.debug(
                f"Figure-rotation check: b/a={b_over_a:.2f} c/a={c_over_a:.2f} "
                f"v_rot/sigma={rotation_measure:.2f} -> not flagged"
            )
        return result

    def estimate_pattern_speed(
        self,
        window_radius: Optional[float] = None,
        significance: float = 3.0,
        check_rigidity: bool = True,
    ) -> dict:
        """Estimate the figure's pattern speed from a single snapshot.

        Uses every field particle, weighted by the smooth radial window
        ``g(r) = exp(-r^2 / (2 R^2))`` with ``R = window_radius``, through the
        shape tensor

            T = sum m r g(r) u u^T,   u = x / r,  r = |x|.

        The snapshot gives both the tensor and its exact instantaneous rate of
        change,

            dT/dt = sum m [(g + r g') v_r u u^T + g (w u^T + u w^T)],

        with ``v_r = u . v`` and ``w = v - u v_r``. Each particle contributes
        ``m`` times a velocity, bounded at both small and large radius. A
        figure rotating rigidly at angular velocity ``Omega`` has
        ``dT/dt = [Omega x, T]``. In the principal frame (eigenvalues
        ``lambda_i``) the off-diagonal rates therefore give every component:

            Omega_k = (dT/dt)_ij / (lambda_i - lambda_j),   (i, j, k) cyclic.

        This is the three-dimensional form of the single-snapshot moment method
        of Dehnen, Semczuk & Schoenrich (2023). It is a continuity-equation
        estimator, and it is unbiased for a figure rotating rigidly and
        steadily because no particle is selected by its velocity (see
        :func:`_windowed_pattern_speed`). The estimate is then ``Omega`` plus
        the rotating-frame streaming term, whose expectation vanishes.
        Streaming in a figure that does not tumble (ordered rotation of the
        stars) therefore gives no signal.

        Each component's Poisson uncertainty is estimated from the scatter of
        the per-particle contributions to ``(dT/dt)_ij``. A component is kept
        only if it exceeds ``significance`` times its uncertainty; otherwise it
        is set to zero. Rotation about an axis of symmetry (``lambda_i`` close
        to ``lambda_j``) is undefined: the uncertainty then diverges and the
        component is dropped. A spherical, axisymmetric or non-tumbling figure
        therefore gets a zero pattern speed, and its orbits are integrated in
        a static potential.

        The estimate assumes the figure rotates rigidly and steadily. With
        ``check_rigidity`` it is compared with :meth:`pattern_speed_profile`,
        and a warning is logged if the shells disagree (differential rotation,
        or a figure that is still changing shape). The system should already be
        recentred in position and velocity (see :meth:`recentre`).

        Parameters
        ----------
        window_radius : float, optional
            Scale ``R`` of the radial window (physical length units). Defaults
            to the field half-mass radius about the origin.
        significance : float, optional
            A component is used only if ``|Omega_k|`` exceeds this many times
            its uncertainty (default 3).
        check_rigidity : bool, optional
            If True (default) and a pattern speed is detected, compute
            :meth:`pattern_speed_profile` and warn if any shell disagrees with
            the estimate by more than ``significance`` times its uncertainty.

        Returns
        -------
        result : dict
            ``"pattern_speed"`` (numpy.ndarray, (3,) angular velocity in the
            current frame, physical units, insignificant components zeroed);
            ``"principal_axes"`` (numpy.ndarray, (3, 3), columns are the long,
            intermediate and short axes of the windowed tensor in the current
            frame); ``"principal_pattern_speed"`` and ``"uncertainty"``
            (numpy.ndarray, (3,) raw estimate and its 1-sigma uncertainty about
            each principal axis); ``"significant"`` (numpy.ndarray of bool,
            (3,)); ``"eigenvalues"`` (numpy.ndarray, (3,) of the normalised
            tensor, descending); ``"window_radius"`` (float); ``"n_particles"``
            (int, particles with non-negligible weight); and ``"profile"``
            (the :meth:`pattern_speed_profile` result, or None if not computed).
        """
        fld = self.field
        result = {
            "pattern_speed": np.zeros(3),
            "principal_axes": np.eye(3),
            "principal_pattern_speed": np.full(3, np.nan),
            "uncertainty": np.full(3, np.nan),
            "significant": np.zeros(3, dtype=bool),
            "eigenvalues": np.full(3, np.nan),
            "window_radius": float("nan"),
            "n_particles": 0,
            "profile": None,
        }
        if fld.n_particles < 100:
            logger.warning(
                "Too few field particles to estimate the pattern speed; "
                "assuming a static figure"
            )
            return result

        if window_radius is None:
            window_radius = self.half_mass_radius()
        window_radius = float(window_radius)
        g, dg_dr = _gaussian_window(np.linalg.norm(fld.pos, axis=1), window_radius)
        omega_p, sigma_p, vals, vecs, n = _windowed_pattern_speed(
            fld.pos, fld.vel, fld.mass, g, dg_dr
        )
        significant = np.isfinite(omega_p) & (np.abs(omega_p) > significance * sigma_p)
        omega = vecs @ np.where(significant, omega_p, 0.0)

        result.update(
            pattern_speed=omega,
            principal_axes=vecs,
            principal_pattern_speed=omega_p,
            uncertainty=sigma_p,
            significant=significant,
            eigenvalues=vals,
            window_radius=window_radius,
            n_particles=n,
        )
        for name, k in (("long", 0), ("intermediate", 1), ("short", 2)):
            logger.debug(
                f"Pattern speed about the {name} axis: {omega_p[k]:.4g} +/- "
                f"{sigma_p[k]:.2g} ({'kept' if significant[k] else 'dropped'})"
            )
        if not significant.any():
            logger.info(
                f"No significant figure rotation detected from {n} field "
                f"particles (window radius {window_radius:.4g}; pattern speed "
                f"set to zero)"
            )
            return result

        logger.info(
            f"Estimated pattern speed {np.round(omega, 4)} "
            f"(|Omega|={np.linalg.norm(omega):.4g}, velocity/length units) from "
            f"{n} field particles (window radius {window_radius:.4g})"
        )
        if check_rigidity:
            profile = self.pattern_speed_profile(
                radii=window_radius * np.array([0.25, 0.5, 1.0, 2.0])
            )
            result["profile"] = profile
            speed = np.linalg.norm(omega)
            axis = omega / speed
            deviant = []
            for radius, omega_s, sigma_s, axes_s in zip(
                profile["radius"],
                profile["pattern_speed"],
                profile["uncertainty"],
                profile["principal_axes"],
            ):
                # Uncertainty of the shell's rate about the estimated axis.
                weights = (axes_s.T @ axis) ** 2
                sigma_along = np.sqrt(
                    np.sum(weights * np.nan_to_num(sigma_s, nan=np.inf) ** 2)
                )
                along = float(omega_s @ axis)
                if (
                    np.isfinite(sigma_along)
                    and abs(along - speed) > significance * sigma_along
                ):
                    deviant.append(f"r={radius:.3g}: {along:.4g} +/- {sigma_along:.2g}")
            if deviant:
                logger.warning(
                    f"The pattern speed varies with radius (estimate {speed:.4g}; "
                    f"{'; '.join(deviant)}): the figure does not rotate rigidly, "
                    f"or is still changing shape, so a single pattern speed is "
                    f"only an average -- see pattern_speed_profile()"
                )
        return result

    def pattern_speed_profile(self, radii=None, width: float = 0.35) -> dict:
        """Pattern speed measured in a set of overlapping radial shells.

        Applies the estimator of :meth:`estimate_pattern_speed` with a smooth
        log-normal shell window ``g(r) = exp(-ln(r / r_s)^2 / (2 width^2))``
        about each radius ``r_s``, in place of the central Gaussian. A rigidly
        rotating figure gives the same pattern speed in every shell (within the
        uncertainties); a trend with radius means differential rotation or a
        figure that is changing shape. Shells where the figure is nearly
        spherical or axisymmetric have large uncertainties, as the rotation is
        undefined there. No significance cut is applied.

        Parameters
        ----------
        radii : array-like of float, optional
            Shell radii (physical length units). Defaults to 0.25, 0.5, 1 and 2
            times the field half-mass radius.
        width : float, optional
            Width of each shell in ``ln r`` (default 0.35, so shells a factor
            of two apart overlap).

        Returns
        -------
        profile : dict
            ``"radius"`` (numpy.ndarray, (n,)); ``"pattern_speed"``
            (numpy.ndarray, (n, 3) raw angular velocity in the current frame);
            ``"principal_pattern_speed"`` and ``"uncertainty"`` (numpy.ndarray,
            (n, 3) about each shell's principal axes); ``"principal_axes"``
            (numpy.ndarray, (n, 3, 3)); and ``"n_particles"`` (numpy.ndarray,
            (n,) particles with non-negligible weight).
        """
        fld = self.field
        if radii is None:
            radii = self.half_mass_radius() * np.array([0.25, 0.5, 1.0, 2.0])
        radii = np.atleast_1d(np.asarray(radii, dtype=np.float64))
        r = np.linalg.norm(fld.pos, axis=1)
        log_r = np.log(np.where(r > 0, r, np.nan))

        n_shells = len(radii)
        omega = np.full((n_shells, 3), np.nan)
        omega_principal = np.full((n_shells, 3), np.nan)
        sigma = np.full((n_shells, 3), np.nan)
        axes = np.tile(np.eye(3), (n_shells, 1, 1))
        counts = np.zeros(n_shells, dtype=np.int64)
        for s, radius in enumerate(radii):
            offset = (log_r - np.log(radius)) / width
            g = np.nan_to_num(np.exp(-0.5 * offset**2))
            dg_dr = np.nan_to_num(-offset / (width * r) * g)
            omega_p, sigma_p, _, vecs, n = _windowed_pattern_speed(
                fld.pos, fld.vel, fld.mass, g, dg_dr
            )
            omega_principal[s], sigma[s], axes[s], counts[s] = omega_p, sigma_p, vecs, n
            omega[s] = vecs @ np.nan_to_num(omega_p)
        return {
            "radius": radii,
            "pattern_speed": omega,
            "principal_pattern_speed": omega_principal,
            "uncertainty": sigma,
            "principal_axes": axes,
            "n_particles": counts,
        }

    def prepare(
        self,
        centre: str = "bh",
        check_figure_rotation: bool = True,
        pattern_speed="estimate",
    ) -> None:
        """Recentre, align, estimate the scale radius and the pattern speed.

        Convenience wrapper that runs :meth:`recentre`, :meth:`align` and
        :meth:`estimate_scale_radius` in sequence, then sets
        :attr:`pattern_speed`, which the potentials built from this system
        inherit. By default the pattern speed is estimated with
        :meth:`estimate_pattern_speed`.

        Parameters
        ----------
        centre : str, optional
            How to define the centre, passed to :meth:`recentre` (default
            ``"bh"``).
        check_figure_rotation : bool, optional
            If True (default) and the resulting pattern speed is zero (so
            orbits will be integrated in a static potential), run
            :meth:`detect_figure_rotation` and log a warning if the figure
            nevertheless looks like it is tumbling.
        pattern_speed : str, float or array-like of float, optional
            ``"estimate"`` (default) to measure it with
            :meth:`estimate_pattern_speed`; ``"none"`` for a static figure; a
            number for rotation about the (aligned) short axis z; or a (3,)
            angular-velocity vector in the aligned frame. Physical units
            (velocity unit / length unit, e.g. km/s/kpc).

        Raises
        ------
        ValueError
            If ``pattern_speed`` is not one of the accepted forms.
        """
        estimate = (
            isinstance(pattern_speed, str) and pattern_speed.lower() == "estimate"
        )
        omega = None if estimate else _as_pattern_speed(pattern_speed)

        self.pattern_speed = None
        self.recentre(on=centre)
        self.align()
        self.estimate_scale_radius()
        if estimate:
            omega = self.estimate_pattern_speed()["pattern_speed"]
        elif np.any(omega):
            logger.info(f"Using the given pattern speed {np.round(omega, 4)}")
        self.pattern_speed = omega
        if check_figure_rotation and not np.any(omega):
            self.detect_figure_rotation()
