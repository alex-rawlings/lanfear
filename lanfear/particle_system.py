"""Particle data container and Gadget-4 HDF5 reader.

A :class:`ParticleSystem` holds positions, velocities, masses, IDs and a
per-particle species label. It knows how to load a Gadget-4 HDF5 snapshot,
recentre and align the system, and hand off the field (non-BH) particles to the
SCF potential in Hernquist-Ostriker (HO) units.

HO units: ``G = M_field = scale_radius = 1``. The scale radius is estimated from
the field half-mass radius as ``r_half / (1 + sqrt(2))`` (exact for a Hernquist
profile), and the mass unit is the total field (non-BH) mass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from ._logging import get_logger

logger = get_logger(__name__)

# Gadget PartType -> species label used throughout the code.
_PARTTYPE_TO_SPECIES = {
    "PartType0": "GAS",
    "PartType1": "DM",
    "PartType2": "DISK",
    "PartType3": "BULGE",
    "PartType4": "STAR",
    "PartType5": "BH",
}


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
    """

    pos: np.ndarray
    vel: np.ndarray
    mass: np.ndarray
    ids: np.ndarray
    species: np.ndarray
    scale_radius: Optional[float] = field(default=None)

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
        )
        logger.info("Loaded %d particles from %s", system.n_particles, filename)
        labels, counts = np.unique(system.species, return_counts=True)
        logger.debug(
            "Species breakdown: %s",
            {str(s): int(c) for s, c in zip(labels, counts)},
        )
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
            current scale radius).
        """
        mask = np.asarray(mask)
        return ParticleSystem(
            pos=self.pos[mask],
            vel=self.vel[mask],
            mass=self.mass[mask],
            ids=self.ids[mask],
            species=self.species[mask],
            scale_radius=self.scale_radius,
        )

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
        logger.info("Estimated scale radius: %.4g", self.scale_radius)
        return self.scale_radius

    # --------------------------------------------------------- preparation
    def recentre(self, on: str = "field") -> None:
        """Shift positions/velocities so the chosen centre is the origin.

        Modifies the system in place.

        Parameters
        ----------
        on : str, optional
            Which subset defines the centre: ``"field"`` (field COM, the
            default), ``"bh"`` (black-hole COM), or a species label such as
            ``"STAR"``.

        Raises
        ------
        ValueError
            If no particles match the requested centre selection.
        """
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
        logger.debug(
            "Recentred on '%s'; shifted position COM by %s", on, np.round(pos_com, 4)
        )

    def align(self) -> np.ndarray:
        """Rotate so the field principal axes align with x, y, z (in place).

        Uses the distance-normalised reduced inertia tensor of the field
        particles; the longest axis maps to x and the shortest to z. Assumes the
        system has already been recentred.

        Returns
        -------
        rotation : numpy.ndarray
            The (3, 3) rotation matrix applied to positions and velocities.
        """
        fld = self.field
        r2 = fld.radii() ** 2
        good = r2 > 0
        p = fld.pos[good]
        m = fld.mass[good]
        w = m / r2[good]
        # Reduced inertia tensor I_ij = sum w * x_i x_j.
        tensor = np.einsum("k,ki,kj->ij", w, p, p)
        vals, vecs = np.linalg.eigh(tensor)
        # Largest eigenvalue -> longest axis -> map to x.
        order = np.argsort(vals)[::-1]
        rot = vecs[:, order].T
        # Ensure a proper rotation (det +1).
        if np.linalg.det(rot) < 0:
            rot[2] *= -1
        self.pos = self.pos @ rot.T
        self.vel = self.vel @ rot.T
        logger.debug("Aligned field principal axes with x, y, z")
        return rot

    def detect_figure_rotation(
        self,
        axis_ratio_threshold: float = 0.9,
        rotation_threshold: float = 0.3,
    ) -> dict:
        """Heuristically flag likely figure rotation (a tumbling figure).

        Figure rotation -- the pattern speed at which a non-axisymmetric figure
        tumbles -- is a *time-dependent* quantity that a single snapshot cannot
        measure rigorously (that needs consecutive snapshots or a
        Tremaine-Weinberg-type analysis). This method instead flags the regime
        in which figure rotation is likely and in which the classifier (which
        integrates orbits in a *static* potential) would mis-assign families: a
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
                "Possible figure rotation: non-axisymmetric field (b/a=%.2f) with "
                "significant ordered rotation about its short axis "
                "(v_rot/sigma=%.2f). The classifier integrates orbits in a STATIC "
                "potential and will mis-assign families if the figure is tumbling; "
                "confirm the pattern speed from consecutive snapshots before "
                "trusting the classification.",
                b_over_a,
                rotation_measure,
            )
        else:
            logger.debug(
                "Figure-rotation check: b/a=%.2f c/a=%.2f v_rot/sigma=%.2f "
                "-> not flagged",
                b_over_a,
                c_over_a,
                rotation_measure,
            )
        return result

    def prepare(
        self, centre: str = "field", check_figure_rotation: bool = True
    ) -> None:
        """Recentre, align, and estimate the scale radius (in place).

        Convenience wrapper that runs :meth:`recentre`, :meth:`align` and
        :meth:`estimate_scale_radius` in sequence, then (optionally) checks for
        figure rotation and warns if it is detected.

        Parameters
        ----------
        centre : str, optional
            Subset defining the centre, passed to :meth:`recentre` (default
            ``"field"``).
        check_figure_rotation : bool, optional
            If True (default), run :meth:`detect_figure_rotation` and log a
            warning if figure rotation is detected.
        """
        self.recentre(on=centre)
        self.align()
        self.estimate_scale_radius()
        if check_figure_rotation:
            self.detect_figure_rotation()
