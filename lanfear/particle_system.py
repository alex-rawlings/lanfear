"""Particle data container and Gadget-3/4 HDF5 reader.

A :class:`ParticleSystem` holds positions, velocities, masses, IDs and a
per-particle species label. It knows how to load a Gadget-3/4 HDF5 snapshot
(whole, or only a list of particle IDs, streamed in chunks so a selection from
a snapshot too large for memory still loads; a selection from a periodic box is
unwrapped across the box edges; everything is converted to physical units),
recentre and align the system, estimate the pattern speed of a tumbling figure,
and hand off the field (non-BH) particles to the SCF potential in
Hernquist-Ostriker (HO) units.

HO units: ``G = M_field = scale_radius = 1``. The scale radius is estimated from
the field half-mass radius as ``r_half / (1 + sqrt(2))`` (exact for a Hernquist
profile), and the mass unit is the total field (non-BH) mass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple

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

# Rows read from the snapshot per chunk by _iter_gadget_hdf5 (~200 MB of
# float64 positions and velocities), which bounds the transient memory of a
# selective load independently of the snapshot size.
_READ_CHUNK = 1 << 22


# Default (a, h) exponents converting a stored Gadget field to physical units,
# for datasets that do not record their own ``a_scaling``/``h_scaling``
# attributes: comoving positions in length/h, velocities stored as
# w = v_pec / sqrt(a), and masses in mass/h.
_PHYSICAL_SCALING = {
    "Coordinates": (1.0, -1.0),
    "Velocities": (0.5, 0.0),
    "Masses": (0.0, -1.0),
}


@dataclass(frozen=True)
class _Cosmology:
    """Scale factor and Hubble parameter of a snapshot.

    Both are 1 for a non-cosmological snapshot, which makes the conversion to
    physical units the identity.

    Parameters
    ----------
    scale_factor : float, optional
        Cosmological scale factor ``a`` of the snapshot.
    hubble_param : float, optional
        Dimensionless Hubble parameter ``h``.
    """

    scale_factor: float = 1.0
    hubble_param: float = 1.0

    def factor(self, name: str, attrs=None) -> float:
        """Factor converting a stored field to physical units.

        Parameters
        ----------
        name : str
            ``"Coordinates"``, ``"Velocities"`` or ``"Masses"``.
        attrs : h5py.AttributeManager, optional
            The dataset's attributes; their ``a_scaling`` and ``h_scaling``
            exponents (written by Gadget-4; Gadget-3 has none) override the
            defaults.

        Returns
        -------
        factor : float
            ``a**a_scaling * h**h_scaling``.
        """
        a_exp, h_exp = _PHYSICAL_SCALING[name]
        if attrs is not None:
            a_exp = float(attrs.get("a_scaling", a_exp))
            h_exp = float(attrs.get("h_scaling", h_exp))
        return self.scale_factor**a_exp * self.hubble_param**h_exp


def _header_is_comoving(header: dict, filename: str) -> bool:
    """Whether a Gadget-2/3 header describes a comoving integration.

    These headers have no ``ComovingIntegrationOn`` flag. A comoving run writes
    ``Redshift = 1 / Time - 1`` and a non-comoving one ``Redshift = 0`` (with
    ``Time`` a time), and cosmological runs have ``Omega0 > 0``; a run is taken
    to be comoving when both hold. ``Time = 1`` with ``Redshift = 0`` fits
    either and is taken as comoving (with a warning).

    Parameters
    ----------
    header : dict
        The ``Header`` attributes.
    filename : str
        Snapshot path, for the log messages.

    Returns
    -------
    comoving : bool
        Whether the snapshot is from a comoving integration.
    """
    a = header.get("Time", None)
    z = header.get("Redshift", None)
    if a is None or z is None or float(a) <= 0.0:
        return False
    if float(header.get("Omega0", 1.0)) <= 0.0:
        return False
    if not np.isclose(float(z), 1.0 / float(a) - 1.0, rtol=1e-6, atol=1e-8):
        return False
    if float(z) == 0.0:
        logger.warning(
            f"{filename}: Time = 1 and Redshift = 0 fit both a comoving run at "
            f"a = 1 and a non-comoving one at t = 1; taking it as comoving (pass "
            f"cosmological=False otherwise)"
        )
    return True


def _read_header(
    filename: str,
    periodic: Optional[bool],
    cosmological: Optional[bool],
) -> Tuple[Optional[np.ndarray], Optional[_Cosmology]]:
    """Read the periodic box size and cosmology of a Gadget-3/4 HDF5 snapshot.

    Parameters
    ----------
    filename : str
        Path to the Gadget-3/4 HDF5 snapshot file.
    periodic : bool, optional
        Whether the snapshot is periodic. ``None`` detects it from the
        ``PERIODIC`` compile-time option Gadget-4 records in the ``Config``
        group. A snapshot without that group (e.g. Gadget-3) is taken to be
        periodic if it is cosmological and has a positive ``BoxSize``.
    cosmological : bool, optional
        Whether the snapshot comes from a comoving integration. ``None``
        detects it from the ``ComovingIntegrationOn`` run-time parameter
        Gadget-4 records in the ``Parameters`` group. A snapshot without that
        group (e.g. Gadget-3) is taken to be comoving if its header has
        ``Redshift = 1 / Time - 1`` and ``Omega0 > 0``.

    Returns
    -------
    box_size : numpy.ndarray or None
        (3,) physical box side lengths from the header ``BoxSize`` (a scalar is
        applied to every axis), or ``None`` if the snapshot is not periodic.
    cosmology : _Cosmology or None
        The scale factor (header ``Time``) and Hubble parameter
        (``HubbleParam``) of a cosmological snapshot, or ``None`` if the
        snapshot is not cosmological.

    Raises
    ------
    ValueError
        If the snapshot is periodic but the header has no positive ``BoxSize``,
        or cosmological but has no ``Time`` or ``HubbleParam``.
    """
    import h5py

    with h5py.File(filename, "r") as f:
        header = dict(f["Header"].attrs) if "Header" in f else {}
        parameters = dict(f["Parameters"].attrs) if "Parameters" in f else None
        config = dict(f["Config"].attrs) if "Config" in f else None
    box = header.get("BoxSize", None)
    if cosmological is None:
        if parameters is not None and "ComovingIntegrationOn" in parameters:
            cosmological = bool(parameters["ComovingIntegrationOn"])
        else:
            cosmological = _header_is_comoving(header, filename)
            logger.debug(
                f"{filename} has no ComovingIntegrationOn parameter; inferred "
                f"cosmological={cosmological} from the header"
            )
    if periodic is None:
        if config is not None:
            periodic = "PERIODIC" in config
        else:
            periodic = bool(
                cosmological
                and box is not None
                and np.all(np.asarray(box, dtype=np.float64) > 0.0)
            )
            logger.debug(
                f"{filename} has no Config group; inferred periodic={periodic}"
            )
    cosmology = None
    if cosmological:
        # Gadget-4 keeps HubbleParam in Parameters; Gadget-3 in the Header.
        a = header.get("Time", None)
        h = (parameters or {}).get("HubbleParam", header.get("HubbleParam", None))
        if a is None or h is None:
            raise ValueError(
                f"Cosmological snapshot {filename} has no Time (scale factor) "
                f"or HubbleParam"
            )
        cosmology = _Cosmology(scale_factor=float(a), hubble_param=float(h))
        logger.info(
            f"Converting to physical units: a = {cosmology.scale_factor:.6g}, "
            f"h = {cosmology.hubble_param:.6g}"
        )
    if not periodic:
        return None, cosmology
    if box is None or np.any(np.asarray(box, dtype=np.float64) <= 0.0):
        raise ValueError(f"Periodic snapshot {filename} has no positive BoxSize")
    box = np.broadcast_to(np.asarray(box, dtype=np.float64), (3,)).copy()
    if cosmology is not None:
        box *= cosmology.factor("Coordinates")
    return box, cosmology


def _unwrap_periodic(pos: np.ndarray, mass: np.ndarray, box_size: np.ndarray) -> None:
    """Make a selection from a periodic box contiguous, in place.

    Each axis is wrapped to within half a box of the mass-weighted circular
    mean of the positions, which, unlike the plain mean, is not dragged
    towards the box centre by particles split across an edge. A system smaller
    than half the box then has no particle on the far side of an edge, so
    distances, centring and the potential fit need no minimum-image
    convention. Positions keep their snapshot frame (modulo the box size).

    Parameters
    ----------
    pos : numpy.ndarray
        (N, 3) positions, modified in place.
    mass : numpy.ndarray
        (N,) masses weighting the circular mean.
    box_size : numpy.ndarray
        (3,) box side lengths.
    """
    phase = 2.0 * np.pi * pos / box_size
    angle = np.arctan2(mass @ np.sin(phase), mass @ np.cos(phase))
    reference = np.mod(angle * box_size / (2.0 * np.pi), box_size)
    pos -= reference
    pos += 0.5 * box_size
    np.mod(pos, box_size, out=pos)
    pos += reference - 0.5 * box_size
    logger.debug(f"Unwrapped the periodic box about {reference}")
    extent = np.ptp(pos, axis=0)
    if np.any(extent > 0.5 * box_size):
        logger.warning(
            f"The particles span {np.round(extent / box_size, 3)} of the box "
            f"along each axis; a system wider than half the box cannot be "
            f"unwrapped unambiguously"
        )


def _read_physical(grp, name: str, start: int, stop: int, keep, to_physical: dict):
    """Read the kept rows of one block of a dataset, in physical units.

    Parameters
    ----------
    grp : h5py.Group
        The ``PartTypeX`` group.
    name : str
        Dataset name.
    start, stop : int
        Row range of the block.
    keep : numpy.ndarray or slice
        Rows of the block to keep.
    to_physical : dict
        Physical-unit conversion factor of each dataset.

    Returns
    -------
    values : numpy.ndarray
        The kept rows as float64, in physical units.
    """
    values = np.asarray(grp[name][start:stop], dtype=np.float64)[keep]
    if to_physical[name] != 1.0:
        values *= to_physical[name]
    return values


def _read_block(
    grp,
    start: int,
    stop: int,
    wanted: Optional[np.ndarray],
    const: Optional[float],
    to_physical: dict,
    label: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Read the selected rows of one block of a particle group.

    The ``ParticleIDs`` of the block are read first, so the other datasets are
    only read for blocks holding a requested particle.

    Parameters
    ----------
    grp : h5py.Group
        The ``PartTypeX`` group.
    start, stop : int
        Row range of the block.
    wanted : numpy.ndarray or None
        Sorted requested IDs, in the group's ID dtype; ``None`` keeps every row.
    const : float or None
        Physical mass of every particle of a group without ``Masses``; ``None``
        if the group has per-particle masses.
    to_physical : dict
        Physical-unit conversion factor of each dataset.
    label : str
        Species label of the group.

    Returns
    -------
    block : tuple of numpy.ndarray or None
        ``(pos, vel, mass, ids, species)`` of the kept rows in physical units,
        or ``None`` if no row of the block was requested.
    """
    if "ParticleIDs" in grp:
        chunk_ids = np.asarray(grp["ParticleIDs"][start:stop])
    else:
        chunk_ids = np.arange(start, stop, dtype=np.int64)
    keep = slice(None)
    if wanted is not None:
        # Membership by binary search in the sorted request.
        at = np.searchsorted(wanted, chunk_ids)
        at[at == len(wanted)] = 0
        keep = np.flatnonzero(wanted[at] == chunk_ids)
        if keep.size == 0:
            return None
    if const is None:
        m = _read_physical(grp, "Masses", start, stop, keep, to_physical)
    else:
        m = np.full(stop - start, const, dtype=np.float64)[keep]
    return (
        _read_physical(grp, "Coordinates", start, stop, keep, to_physical),
        _read_physical(grp, "Velocities", start, stop, keep, to_physical),
        m,
        chunk_ids[keep],
        np.full(len(m), label),
    )


# State of a parallel-reader worker process, set once per process by
# _init_read_worker: its own read-only handle on the snapshot and each group's
# requested IDs (sent once per worker rather than with every block).
_READ_WORKER_STATE: dict = {}


def _init_read_worker(filename: str, wanted: dict) -> None:
    """Open the snapshot in a parallel-reader worker process.

    Parameters
    ----------
    filename : str
        Path to the snapshot.
    wanted : dict
        Sorted requested IDs (or ``None``) of each ``PartTypeX`` group.
    """
    import h5py

    _READ_WORKER_STATE["file"] = h5py.File(filename, "r")
    _READ_WORKER_STATE["wanted"] = wanted


def _read_block_in_worker(key, start, stop, const, to_physical, label):
    """Read one block in a parallel-reader worker; see :func:`_read_block`."""
    grp = _READ_WORKER_STATE["file"][key]
    wanted = _READ_WORKER_STATE["wanted"][key]
    return _read_block(grp, start, stop, wanted, const, to_physical, label)


def _iter_blocks_parallel(
    filename: str, tasks: list, wanted: dict, n_workers: int
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Read blocks in worker processes, yielding them in snapshot order.

    HDF5 is not thread-safe and h5py serialises every call behind one lock, so
    threads would read one block at a time; separate processes, each with its
    own read-only file handle, read concurrently. They share no mutable state:
    each returns new arrays, and the results are yielded in task order
    whatever order they finish in, so the output matches a serial read.
    Workers are spawned rather than forked, as a fork copies any lock held by
    another thread of the parent (HDF5's, OpenMP's, MPI's) into the child,
    where it is never released. At most ``2 * n_workers`` blocks are in
    flight, which bounds the memory held by finished but unconsumed blocks.

    Parameters
    ----------
    filename : str
        Path to the snapshot.
    tasks : list of tuple
        ``(key, start, stop, const, to_physical, label)`` of each block, in
        snapshot order.
    wanted : dict
        Sorted requested IDs (or ``None``) of each group.
    n_workers : int
        Number of worker processes.

    Yields
    ------
    pos, vel, mass, ids, species : numpy.ndarray
        As in :func:`_iter_gadget_hdf5`.
    """
    import itertools
    import multiprocessing
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor

    todo = iter(tasks)
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_read_worker,
        initargs=(filename, wanted),
    ) as pool:
        pending = deque(
            pool.submit(_read_block_in_worker, *task)
            for task in itertools.islice(todo, 2 * n_workers)
        )
        try:
            while pending:
                block = pending.popleft().result()
                task = next(todo, None)
                if task is not None:
                    pending.append(pool.submit(_read_block_in_worker, *task))
                if block is not None:
                    yield block
        finally:
            # Abandoned early (consumer stopped or a worker raised): drop the
            # queued blocks; the pool then waits for the running ones.
            for future in pending:
                future.cancel()


def _iter_gadget_hdf5(
    filename: str,
    ids: Optional[np.ndarray] = None,
    chunk_size: int = _READ_CHUNK,
    cosmology: Optional[_Cosmology] = None,
    n_workers: int = 1,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield the particles of a Gadget-3/4 HDF5 snapshot chunk by chunk.

    Each ``PartTypeX`` group with ``Coordinates`` is read in blocks of
    ``chunk_size`` rows. If ``ids`` is given, the ``ParticleIDs`` of a block
    are read first and only the matching rows are kept, so the full snapshot is
    never held in memory: the cost is one block per reader plus the selected
    particles. Positions, velocities and masses are converted to physical
    units. With ``n_workers > 1`` the blocks are read by that many processes
    (see :func:`_iter_blocks_parallel`), with the same output.

    Parameters
    ----------
    filename : str
        Path to the Gadget-3/4 HDF5 snapshot file.
    ids : numpy.ndarray, optional
        Sorted, unique particle IDs to keep. ``None`` keeps every particle.
    chunk_size : int, optional
        Rows read per block.
    cosmology : _Cosmology, optional
        Scale factor and Hubble parameter converting the stored values to
        physical units; ``None`` (default) for a non-cosmological snapshot,
        which is already physical.
    n_workers : int, optional
        Number of processes reading blocks (default 1: read serially).

    Yields
    ------
    pos, vel, mass, ids, species : numpy.ndarray
        The selected rows of one block (blocks without any are skipped), in
        physical units and snapshot order, as in :class:`ParticleSystem`.
    """
    import h5py

    if cosmology is None:
        cosmology = _Cosmology()
    tasks, wanted = [], {}
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
            label = _PARTTYPE_TO_SPECIES.get(key, key)
            has_ids = "ParticleIDs" in grp
            const = None
            if "Masses" not in grp:
                ptype = int(key.replace("PartType", ""))
                const = mass_table[ptype] if mass_table is not None else 0.0
                const *= cosmology.factor("Masses")
            to_physical = {
                name: cosmology.factor(name, grp[name].attrs)
                for name in ("Coordinates", "Velocities", "Masses")
                if name in grp
            }
            wanted[key] = None
            if ids is not None:
                # Compare in the snapshot's own ID dtype (often uint32/uint64),
                # dropping requested IDs that dtype cannot hold.
                cast = ids.astype(grp["ParticleIDs"].dtype if has_ids else np.int64)
                wanted[key] = np.sort(cast[cast.astype(ids.dtype) == ids])
            for start in range(0, n, chunk_size):
                stop = min(start + chunk_size, n)
                tasks.append((key, start, stop, const, to_physical, label))

        n_workers = max(1, min(n_workers, len(tasks)))
        if n_workers == 1:
            for key, start, stop, const, to_physical, label in tasks:
                block = _read_block(
                    f[key], start, stop, wanted[key], const, to_physical, label
                )
                if block is not None:
                    yield block
            return
    # The parent's handle is closed before the workers open their own.
    yield from _iter_blocks_parallel(filename, tasks, wanted, n_workers)


def _as_pattern_speed(value) -> np.ndarray:
    """Normalise a user-supplied pattern speed to an angular-velocity vector.

    Parameters
    ----------
    value : None, str, float or array-like of float
        ``None`` or ``"none"`` (any case) for no figure rotation; a scalar for
        rotation about the z axis;
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


def _rotate_uncertainty(rotation: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Per-component uncertainty of a rotated vector with independent errors.

    For ``y = R x`` with independent component errors ``sigma_x``,
    ``sigma_y,i = sqrt(sum_j R_ij^2 sigma_x,j^2)``. A NaN (undefined) input
    component propagates to every output component it contributes to.

    Parameters
    ----------
    rotation : numpy.ndarray
        (3, 3) rotation matrix ``R``.
    sigma : numpy.ndarray
        (3,) 1-sigma uncertainties of the components of ``x``.

    Returns
    -------
    sigma_rotated : numpy.ndarray
        (3,) 1-sigma uncertainties of the components of ``R x``.
    """
    weights = np.asarray(rotation, dtype=np.float64) ** 2
    sigma_sq = np.asarray(sigma, dtype=np.float64) ** 2
    # Exactly-zero weights must not turn an undefined component into NaN.
    terms = np.where(weights > 0.0, weights * sigma_sq[None, :], 0.0)
    return np.sqrt(terms.sum(axis=1))


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
    pattern_speed_sigma : numpy.ndarray, optional
        (3,) 1-sigma uncertainty of each component of :attr:`pattern_speed`
        (same units and frame), propagated from the estimator's per-principal-
        axis uncertainties assuming independent errors. Set by :meth:`prepare`
        when the pattern speed is estimated, including when every component was
        below the significance cut and the pattern speed was set to zero (the
        uncertainty then bounds the undetected rotation). ``None`` when the
        pattern speed was given explicitly or not set. NaN components are
        undefined (rotation about an axis of symmetry).
    orientation : str, optional
        The reference :meth:`orient_axes` fixed the axis signs by:
        ``"pattern"`` (+z along the pattern speed) or ``"spin"`` (+z along the
        net angular momentum), and +x along the net ``L_x`` in both cases.
        ``None`` if the axis signs were not fixed (they are then arbitrary).
    box_size : numpy.ndarray, optional
        (3,) side lengths of the periodic box the particles were loaded from,
        along the snapshot's axes; populated by :meth:`from_gadget_hdf5`, which
        has already unwrapped the positions across the box edges. ``None`` for
        a non-periodic snapshot.
    scale_factor : float, optional
        Cosmological scale factor of the snapshot the particles were loaded
        from; populated by :meth:`from_gadget_hdf5`. ``None`` for a
        non-cosmological snapshot.
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
    pattern_speed_sigma: Optional[np.ndarray] = field(default=None)
    orientation: Optional[str] = field(default=None)
    box_size: Optional[np.ndarray] = field(default=None)
    scale_factor: Optional[float] = field(default=None)

    # ------------------------------------------------------------------ IO
    @classmethod
    def from_gadget_hdf5(
        cls,
        filename: str,
        ids=None,
        chunk_size: int = _READ_CHUNK,
        periodic: Optional[bool] = None,
        cosmological: Optional[bool] = None,
        n_workers: int = 1,
    ) -> "ParticleSystem":
        """Load a Gadget-3/4 HDF5 snapshot, optionally only the given particle IDs.

        Reads every ``PartTypeX`` group present, taking ``Coordinates``,
        ``Velocities``, ``ParticleIDs`` and masses (per-particle ``Masses`` if
        present, otherwise the constant from the header ``MassTable``).

        Everything is converted to physical units on loading. For a
        cosmological snapshot (scale factor ``a`` = header ``Time``, Hubble
        parameter ``h`` = ``HubbleParam``) each field is multiplied by
        ``a**a_scaling * h**h_scaling``, with the exponents Gadget-4 records on
        each dataset. Gadget-3 records none, so the defaults apply: positions
        comoving in length/h, ``(1, -1)``; velocities stored as
        ``v_pec / sqrt(a)``, ``(0.5, 0)``; masses in mass/h, ``(0, -1)``.
        Velocities are the physical peculiar velocities,
        without the Hubble flow. A non-cosmological snapshot is already
        physical (``Time`` is then a time and ``HubbleParam`` plays no part in
        the dynamics), so it is left as stored. Units stay those of the
        simulation, e.g. kpc, 1e10 Msun and km/s for the Gadget defaults.

        The snapshot is streamed in blocks of ``chunk_size`` rows. With ``ids``,
        only the matching particles are kept from each block, so a selection
        from a snapshot of billions of particles (e.g. one halo of a
        cosmological box) needs memory for the selection plus one block, not
        the whole snapshot.

        With ``n_workers > 1`` the blocks are read concurrently by that many
        worker processes, each with its own read-only handle on the file, and
        reassembled in snapshot order, so the result is identical to a serial
        read. Threads would not help, as h5py serialises every HDF5 call. Each
        worker holds up to one block and up to ``2 * n_workers`` read blocks
        await assembly, so peak memory grows by about ``3 * n_workers`` blocks.
        The workers are spawned (they import lanfear afresh, about a second),
        so this pays off for large snapshots, especially on parallel or
        network file systems and with compressed datasets. It helps most with
        ``ids``, when the workers send back only the selected rows; a full
        load also copies every particle from the workers to this process.

        For a periodic snapshot the positions are unwrapped across the box
        edges, so a system straddling an edge (e.g. a halo near the boundary of
        a cosmological box) comes out contiguous. Each axis is wrapped to
        within half a box of the mass-weighted circular mean of the loaded
        particles, so the selection must be smaller than half the box (a
        warning is logged otherwise). Velocities are unchanged. The box size
        is converted to physical units like the positions.

        Parameters
        ----------
        filename : str
            Path to the Gadget-3/4 HDF5 snapshot file.
        ids : array_like, optional
            Particle IDs to load: values of the ``ParticleIDs`` datasets, not
            positions in the per-type arrays (which halo finders such as Caesar
            often store). Any species; include the black holes' IDs to centre
            on them. Duplicates are ignored and the particles keep
            their snapshot order. ``None`` (default) loads every particle.
        chunk_size : int, optional
            Rows read per block (default ``2**22``).
        periodic : bool, optional
            Whether the snapshot is a periodic box, with side lengths from the
            header ``BoxSize``. ``None`` (default) detects it from the
            ``PERIODIC`` option Gadget-4 records in the ``Config`` group. A
            snapshot without that group (e.g. Gadget-3) is taken to be periodic
            if it is cosmological and has a positive ``BoxSize``.
        cosmological : bool, optional
            Whether the snapshot comes from a comoving integration and needs
            converting to physical units. ``None`` (default) detects it from
            the ``ComovingIntegrationOn`` parameter Gadget-4 records in the
            ``Parameters`` group. A snapshot without that group (e.g. Gadget-3)
            is taken to be comoving if its header has
            ``Redshift = 1 / Time - 1`` and ``Omega0 > 0``, as Gadget-2/3 write
            for comoving runs (``Redshift = 0`` otherwise). The detected
            setting is logged at debug level; pass ``True``/``False`` to
            override it.
        n_workers : int, optional
            Number of processes reading blocks concurrently (default 1: read
            serially in this process). Capped at the number of blocks.

        Returns
        -------
        system : ParticleSystem
            All (selected) particle groups concatenated into a single system.

        Raises
        ------
        ValueError
            If the file contains no particle group with ``Coordinates``, if
            ``ids`` is given but empty, or if none of ``ids`` is in the
            snapshot, if the snapshot is periodic but has no positive
            ``BoxSize``, or if it is cosmological but has no ``Time`` or
            ``HubbleParam``.
        """
        box_size, cosmology = _read_header(filename, periodic, cosmological)
        if ids is not None:
            ids = np.unique(np.asarray(ids).ravel())
            if ids.size == 0:
                raise ValueError("ids is empty; pass None to load every particle")
        chunks = list(
            _iter_gadget_hdf5(
                filename,
                ids=ids,
                chunk_size=chunk_size,
                cosmology=cosmology,
                n_workers=n_workers,
            )
        )

        if not chunks:
            if ids is not None:
                raise ValueError(
                    f"None of the {ids.size} requested IDs are in {filename}"
                )
            raise ValueError(f"No particle groups with Coordinates in {filename}")

        pos, vel, mass, pids, species = zip(*chunks)
        system = cls(
            pos=np.concatenate(pos),
            vel=np.concatenate(vel),
            mass=np.concatenate(mass),
            ids=np.concatenate(pids),
            species=np.concatenate(species),
            source_file=str(filename),
            box_size=box_size,
            scale_factor=None if cosmology is None else cosmology.scale_factor,
        )
        logger.info(f"Loaded {system.n_particles} particles from {filename}")
        if box_size is not None:
            _unwrap_periodic(system.pos, system.mass, box_size)
        if ids is not None and system.n_particles < ids.size:
            logger.warning(
                f"{ids.size - system.n_particles} of the {ids.size} requested IDs "
                f"are not in {filename}"
            )
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
            current scale radius, source file, centring, pattern speed and its
            uncertainty, orientation, box size and scale factor).
        """
        # np.asarray would turn a slice into an unusable object array.
        if not isinstance(mask, slice):
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
            pattern_speed_sigma=self.pattern_speed_sigma,
            orientation=self.orientation,
            box_size=self.box_size,
            scale_factor=self.scale_factor,
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
            (and to :attr:`pattern_speed` and :attr:`pattern_speed_sigma`, if
            already set).
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
        if self.pattern_speed_sigma is not None:
            self.pattern_speed_sigma = _rotate_uncertainty(
                rot, self.pattern_speed_sigma
            )
        # The new axes have arbitrary signs again (see orient_axes).
        self.orientation = None
        logger.debug(
            f"Aligned field principal axes with x, y, z using {fld.n_particles} "
            f"field particles (window radius {float(window_radius):.4g})"
        )
        return rot

    def _windowed_angular_momentum(self, window_radius: float):
        """Net angular momentum of the field particles in a smooth window.

        Parameters
        ----------
        window_radius : float
            Scale of the Gaussian radial window (physical length units).

        Returns
        -------
        total : numpy.ndarray
            (3,) window-weighted ``sum m g(r) (x x v)``.
        sigma : numpy.ndarray
            (3,) its Poisson uncertainty from the particle-to-particle scatter.
        """
        fld = self.field
        window, _ = _gaussian_window(np.linalg.norm(fld.pos, axis=1), window_radius)
        contribution = (fld.mass * window)[:, None] * np.cross(fld.pos, fld.vel)
        n = len(contribution)
        sigma = (
            np.sqrt(n * np.var(contribution, axis=0, ddof=1))
            if n > 1
            else np.full(3, np.inf)
        )
        return contribution.sum(axis=0), sigma

    def orient_axes(
        self,
        reference: Optional[str] = "auto",
        window_radius: Optional[float] = None,
        significance: float = 3.0,
    ) -> dict:
        """Fix the signs of the aligned axes by half-turn rotations.

        :meth:`align` makes x, y, z the long, intermediate and short axes but
        leaves the sign of each arbitrary, so "prograde" would mean different
        things from run to run. This fixes them, keeping the frame
        right-handed:

        1. a half turn about x (``y, z -> -y, -z``) if the z reference is
           negative: the pattern speed's z component (``"pattern"``) or the
           field's net ``L_z`` (``"spin"``). It leaves ``L_x`` unchanged.
        2. a half turn about z (``x, y -> -x, -y``) if the net ``L_x`` is
           negative. It leaves ``L_z`` unchanged.

        Afterwards a positive ``L_z`` (and a positive azimuthal frequency about
        z) means rotating with the pattern, or with the net spin, and a
        positive ``L_x`` means rotating with any minor-axis rotation. Positions,
        velocities, :attr:`pattern_speed` and :attr:`pattern_speed_sigma` are
        rotated together. The net angular momenta use all field particles
        in the smooth radial window of :meth:`align`.

        Parameters
        ----------
        reference : {"auto", "pattern", "spin"} or None, optional
            What +z points along. ``"auto"`` (default) uses the pattern speed
            if it has a non-zero z component and the net spin otherwise. None
            leaves the axes as they are.
        window_radius : float, optional
            Scale of the Gaussian window (physical length units). Defaults to
            the field half-mass radius.
        significance : float, optional
            A net ``L_z`` (spin reference) or ``L_x`` below this many times its
            Poisson uncertainty is reported as insignificant, with a warning:
            its sign, and so the orientation of that axis, is then arbitrary.

        Returns
        -------
        result : dict
            ``"rotation"`` (numpy.ndarray, (3, 3) proper rotation applied),
            ``"reference"`` (str or None, the z reference used),
            ``"net_angular_momentum"`` and ``"uncertainty"`` (numpy.ndarray,
            (3,) windowed net ``L`` after orienting and its uncertainty), and
            ``"significant"`` (dict with bool entries ``"z"`` and ``"x"``).

        Raises
        ------
        ValueError
            If ``reference`` is not recognised, or is ``"pattern"`` without a
            pattern speed about z.
        """
        if reference not in ("auto", "pattern", "spin", None):
            raise ValueError(
                f"reference must be 'auto', 'pattern', 'spin' or None, got {reference!r}"
            )
        rotation = np.eye(3)
        if reference is None:
            self.orientation = None
            return {
                "rotation": rotation,
                "reference": None,
                "net_angular_momentum": None,
                "uncertainty": None,
                "significant": {"z": False, "x": False},
            }
        omega_z = 0.0 if self.pattern_speed is None else float(self.pattern_speed[2])
        if reference == "auto":
            reference = "pattern" if omega_z != 0.0 else "spin"
        if reference == "pattern" and omega_z == 0.0:
            raise ValueError("reference='pattern' needs a pattern speed about z")
        if window_radius is None:
            window_radius = self.half_mass_radius()
        window_radius = float(window_radius)

        def half_turn(axis):
            flip = np.ones(3)
            flip[[a for a in range(3) if a != axis]] = -1.0
            return np.diag(flip)

        def apply(rot):
            self.pos = self.pos @ rot.T
            self.vel = self.vel @ rot.T
            if self.pattern_speed is not None:
                self.pattern_speed = rot @ self.pattern_speed
            # Sign flips leave per-component uncertainties unchanged.

        total, sigma = self._windowed_angular_momentum(window_radius)
        z_value = omega_z if reference == "pattern" else total[2]
        significant_z = (
            reference == "pattern" or abs(total[2]) > significance * sigma[2]
        )
        if z_value < 0:
            rot = half_turn(0)
            apply(rot)
            rotation = rot @ rotation
            total = rot @ total
        significant_x = bool(abs(total[0]) > significance * sigma[0])
        if total[0] < 0:
            rot = half_turn(2)
            apply(rot)
            rotation = rot @ rotation
            total = rot @ total
        self.orientation = reference
        for name, ok in (("z (net spin)", significant_z), ("x", significant_x)):
            if not ok:
                logger.warning(
                    f"The net L_{name[0]} of the field is consistent with zero "
                    f"(< {significance:g} sigma in the window of radius "
                    f"{window_radius:.4g}); the sign of the {name} axis, and so "
                    f"prograde/retrograde about it, is arbitrary"
                )
        logger.debug(
            f"Oriented axes by {reference!r}: net L = {np.round(total, 4)} "
            f"+/- {np.round(sigma, 4)}"
        )
        return {
            "rotation": rotation,
            "reference": reference,
            "net_angular_momentum": total,
            "uncertainty": sigma,
            "significant": {"z": bool(significant_z), "x": significant_x},
        }

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
        shape tensor::

            T = sum m r g(r) u u^T,   u = x / r,  r = |x|.

        The snapshot gives both the tensor and its exact instantaneous rate of
        change::

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
            each principal axis); ``"pattern_speed_uncertainty"``
            (numpy.ndarray, (3,) 1-sigma uncertainty of each component of
            ``"pattern_speed"`` in the current frame, propagated from
            ``"uncertainty"`` assuming independent errors and kept for dropped
            components too, so it bounds an undetected rotation; NaN where
            undefined); ``"significant"`` (numpy.ndarray of bool,
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
            "pattern_speed_uncertainty": np.full(3, np.nan),
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
            pattern_speed_uncertainty=_rotate_uncertainty(vecs, sigma_p),
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
        orient: Optional[str] = "auto",
    ) -> None:
        """Recentre, align, estimate the scale radius and the pattern speed.

        Convenience wrapper that runs :meth:`recentre`, :meth:`align` and
        :meth:`estimate_scale_radius` in sequence, then sets
        :attr:`pattern_speed`, which the potentials built from this system
        inherit, and finally fixes the axis signs with :meth:`orient_axes`. By
        default the pattern speed is estimated with
        :meth:`estimate_pattern_speed`, and its uncertainty is stored as
        :attr:`pattern_speed_sigma` (``None`` for a given pattern speed).

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
            number for rotation about the short axis, positive for rotation
            in the positive sense about the input frame's +z; or a (3,)
            angular-velocity vector in the input frame. Physical units
            (velocity unit / length unit, e.g. km/s/kpc). A given value is
            rotated with the particles by :meth:`align` and
            :meth:`orient_axes` (so components may change sign), which keeps
            its sense relative to the matter: :meth:`align` picks the sign
            of each axis arbitrarily, so a value in the aligned frame would
            have an arbitrary sense. A number is kept exactly along the
            aligned short axis; a warning is logged if the input z axis is
            more than 60 degrees from it (the sense is then ill-defined).
        orient : {"auto", "pattern", "spin"} or None, optional
            Reference passed to :meth:`orient_axes` (default ``"auto"``: +z
            along the pattern speed if the figure rotates, else along the net
            spin). None leaves the axis signs arbitrary.

        Raises
        ------
        ValueError
            If ``pattern_speed`` is not one of the accepted forms.
        """
        estimate = (
            isinstance(pattern_speed, str) and pattern_speed.lower() == "estimate"
        )
        omega = None if estimate else _as_pattern_speed(pattern_speed)
        about_short_axis = not estimate and np.ndim(pattern_speed) == 0

        self.pattern_speed = None
        self.pattern_speed_sigma = None
        self.recentre(on=centre)
        rotation = self.align()
        self.estimate_scale_radius()
        sigma = None
        if estimate:
            result = self.estimate_pattern_speed()
            omega = result["pattern_speed"]
            sigma = result["pattern_speed_uncertainty"]
        elif np.any(omega):
            # Carry the given value (input frame) through align()'s rotation.
            if about_short_axis:
                # rotation[2, 2] is the cosine between the input and aligned z.
                if abs(rotation[2, 2]) < 0.5:
                    logger.warning(
                        f"The input z axis is {np.degrees(np.arccos(abs(rotation[2, 2]))):.0f} "
                        f"degrees from the short axis, so the sense of the given pattern "
                        f"speed about it is ill-defined; give an angular-velocity vector instead"
                    )
                omega = np.sign(rotation[2, 2]) * omega
            else:
                omega = rotation @ omega
            logger.info(
                f"Using the given pattern speed {np.round(omega, 4)} (aligned frame)"
            )
        self.pattern_speed = omega
        self.pattern_speed_sigma = sigma
        self.orientation = None
        self.orient_axes(orient)
        if check_figure_rotation and not np.any(omega):
            self.detect_figure_rotation()
