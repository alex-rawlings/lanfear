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

    def prepare(self, centre: str = "field") -> None:
        """Recentre, align, and estimate the scale radius (in place).

        Convenience wrapper that runs :meth:`recentre`, :meth:`align` and
        :meth:`estimate_scale_radius` in sequence.

        Parameters
        ----------
        centre : str, optional
            Subset defining the centre, passed to :meth:`recentre` (default
            ``"field"``).
        """
        self.recentre(on=centre)
        self.align()
        self.estimate_scale_radius()
