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
from typing import Optional

import numpy as np

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

    All arrays are indexed consistently. ``pos``/``vel`` have shape ``(N, 3)``;
    ``mass``/``ids``/``species`` have shape ``(N,)``.
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

        return cls(
            pos=np.concatenate(pos),
            vel=np.concatenate(vel),
            mass=np.concatenate(mass),
            ids=np.concatenate(ids),
            species=np.concatenate(species),
        )

    # -------------------------------------------------------------- slicing
    @property
    def n_particles(self) -> int:
        return len(self.mass)

    def select(self, mask) -> "ParticleSystem":
        """Return a new ParticleSystem containing the masked particles."""
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
        """Boolean mask selecting the given species labels (e.g. "STAR")."""
        want = set(labels)
        return np.array([s in want for s in self.species])

    @property
    def field(self) -> "ParticleSystem":
        """The field particles: everything except black holes."""
        return self.select(self.species != "BH")

    @property
    def black_holes(self) -> "ParticleSystem":
        return self.select(self.species == "BH")

    # ----------------------------------------------------------- geometry
    def radii(self, centre=None) -> np.ndarray:
        centre = np.zeros(3) if centre is None else np.asarray(centre)
        return np.linalg.norm(self.pos - centre, axis=1)

    def centre_of_mass(self, species: Optional[str] = None):
        """Mass-weighted centre of mass, optionally of one species."""
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
        """Spherical half-(field-)mass radius about the current origin."""
        fld = self.field
        r = fld.radii()
        order = np.argsort(r)
        cumulative = np.cumsum(fld.mass[order])
        half = cumulative[-1] / 2.0
        idx = np.searchsorted(cumulative, half)
        return float(r[order][min(idx, len(r) - 1)])

    def estimate_scale_radius(self) -> float:
        """Hernquist scale radius estimate: r_half / (1 + sqrt(2))."""
        self.scale_radius = self.half_mass_radius() / (1.0 + np.sqrt(2.0))
        return self.scale_radius

    # --------------------------------------------------------- preparation
    def recentre(self, on: str = "field") -> None:
        """Shift positions/velocities so the chosen centre is the origin.

        ``on`` may be ``"field"`` (field COM), ``"bh"`` (black-hole COM), or a
        species label such as ``"STAR"``.
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

    def align(self) -> np.ndarray:
        """Rotate so the field principal axes align with x, y, z.

        Uses the (distance-normalised) reduced inertia tensor of the field
        particles. Returns the applied rotation matrix. Assumes the system has
        already been recentred.
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
        return rot

    def prepare(self, centre: str = "field") -> None:
        """Recentre, align, and estimate the scale radius (in place)."""
        self.recentre(on=centre)
        self.align()
        self.estimate_scale_radius()
