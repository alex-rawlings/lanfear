"""Orbit integration driver with MPI (mpi4py) particle decomposition.

Orbit integration is embarrassingly parallel across particles, so the work is
split over MPI ranks: rank ``root`` holds the potential and the initial
conditions, broadcasts the (compact) potential to every rank, scatters the
initial states, each rank integrates its share with the C++/OpenMP batch
routine, and the per-orbit summaries are gathered back to ``root``.

The module runs unchanged without MPI: if mpi4py is unavailable or the job has a
single rank, it integrates everything locally. Launch a parallel run with e.g.::

    srun -n 64 python my_analysis.py      # SLURM
    mpirun -n 64 python my_analysis.py

and set ``OMP_NUM_THREADS`` for per-rank threading (hybrid MPI+OpenMP).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

from . import _core
from ._logging import get_logger
from .particle_system import ParticleSystem
from .potential import Potential

logger = get_logger(__name__)

SUMMARY_COLUMNS = list(_core.summary_columns())
_COL_INDEX = {name: i for i, name in enumerate(SUMMARY_COLUMNS)}

# Tag written into OrbitResults.save() archives so load() can validate them.
_RESULTS_FORMAT = "lanfear.OrbitResults"


@dataclass
class OrbitResults:
    """Per-orbit integration/analysis results (populated on the root rank only).

    ``fundamentals`` and ``lines`` carry the frequency data classification
    needs; :func:`analyse_family` / :func:`analyse_states` always populate them,
    and they are ``None`` only for results constructed directly without a
    frequency analysis. All frequencies are signed angular frequencies in HO
    units (rad / HO time); a
    negative sign encodes the sense of circulation. Multiply by ``1 / time_unit``
    for physical angular frequency.

    Parameters
    ----------
    ids : numpy.ndarray
        (N,) particle IDs.
    summary : numpy.ndarray
        (N, len(SUMMARY_COLUMNS)) per-orbit summary rows.
    columns : sequence of str
        Column names for ``summary`` (see :data:`SUMMARY_COLUMNS`).
    time_unit : float
        HO-time -> physical-time conversion factor.
    n_periods : int
        Number of orbital periods integrated.
    n_samples : int
        Number of samples per orbit.
    fundamentals : numpy.ndarray, optional
        (N, 3) signed fundamental frequency per axis (HO units); analysis only.
    lines : numpy.ndarray, optional
        (N, 3, n_lines, 2) leading (freq, amp) spectral lines; analysis only.
    length_unit : float
        HO-length -> physical-length conversion factor (the scale radius). A
        radius in HO units becomes physical when multiplied by this.
    initial_radius : numpy.ndarray
        (N,) instantaneous radius of each orbit at integration start, i.e. the
        snapshot radius from the galaxy centre, in HO units.
    """

    ids: np.ndarray  # (N,) particle IDs
    summary: np.ndarray  # (N, len(SUMMARY_COLUMNS))
    columns: Sequence[str]  # column names for `summary`
    time_unit: float  # HO time -> physical time factor
    length_unit: float  # HO length -> physical length factor (scale radius)
    n_periods: int
    n_samples: int
    initial_radius: np.ndarray  # (N,) snapshot radius, HO units
    fundamentals: Optional[np.ndarray] = None  # (N, 3) leading freq per axis
    lines: Optional[np.ndarray] = None  # (N, 3, n_lines, 2) freq, amp

    def column(self, name: str) -> np.ndarray:
        """Return the named summary column.

        Parameters
        ----------
        name : str
            A column name from :data:`SUMMARY_COLUMNS`.

        Returns
        -------
        values : numpy.ndarray
            (N,) values of that column.
        """
        return self.summary[:, _COL_INDEX[name]]

    @property
    def period_physical(self) -> np.ndarray:
        """Estimated orbital period of each particle in physical time units.

        Returns
        -------
        period : numpy.ndarray
            (N,) orbital periods in physical units.
        """
        return self.column("period") * self.time_unit

    @property
    def ok(self) -> np.ndarray:
        """Boolean mask of orbits that integrated successfully.

        Returns
        -------
        mask : numpy.ndarray
            (N,) True where the integration status is 0.
        """
        return self.column("status") == 0

    @property
    def frequency_ratios(self) -> np.ndarray:
        """Ratios of the fundamental frequencies (a classification building block).

        Returns
        -------
        ratios : numpy.ndarray
            (N, 2) ``|w_x|/|w_z|`` and ``|w_y|/|w_z|``; NaN where a denominator
            vanishes (e.g. an axis with no oscillation).

        Raises
        ------
        ValueError
            If no frequency data is present (use ``analyse_family``/
            ``analyse_states``).
        """
        if self.fundamentals is None:
            raise ValueError("no frequency data; use analyse_family/analyse_states")
        w = np.abs(self.fundamentals)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.stack([w[:, 0] / w[:, 2], w[:, 1] / w[:, 2]], axis=1)

    def to_dict(self) -> dict:
        """Flatten the results into a ``name -> array`` dictionary.

        Returns
        -------
        data : dict
            One entry per summary column plus ``"id"`` and, when frequency data
            is present, ``"freq_x"``/``"freq_y"``/``"freq_z"``.
        """
        d = {name: self.summary[:, i] for i, name in enumerate(self.columns)}
        d["id"] = self.ids
        if self.fundamentals is not None:
            d["freq_x"] = self.fundamentals[:, 0]
            d["freq_y"] = self.fundamentals[:, 1]
            d["freq_z"] = self.fundamentals[:, 2]
        return d

    def save(self, path: Union[str, os.PathLike]) -> str:
        """Write these results to a compact ``.npz`` file.

        Orbit integration is expensive, so this saves everything needed to
        rebuild the :class:`OrbitResults` (per-orbit summary, IDs, column names,
        the integration metadata, and the frequency data when present) into a
        single NumPy archive. Reload it with :meth:`load` to resume analysis
        (classification, plotting) without re-integrating.

        The archive is *uncompressed*, and the large per-orbit float arrays
        (``summary``, ``fundamentals``, ``lines``, ``initial_radius``) are
        narrowed to float32: this data barely compresses (it is float64
        mantissas), so DEFLATE was paying full CPU cost on both save and load
        for close to no size benefit, while narrowing the dtype shrinks the
        file *and* removes that cost -- both faster to write and, more
        importantly, much faster for :meth:`load` to read back.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file. A ``.npz`` suffix is appended by NumPy if absent.

        Returns
        -------
        path : str
            The path written (with the ``.npz`` suffix NumPy uses).
        """
        arrays = {
            "_format": np.asarray(_RESULTS_FORMAT),
            "ids": np.asarray(self.ids),
            "summary": np.asarray(self.summary, dtype=np.float32),
            "columns": np.asarray(list(self.columns)),
            "time_unit": np.asarray(self.time_unit, dtype=np.float64),
            "length_unit": np.asarray(self.length_unit, dtype=np.float64),
            "n_periods": np.asarray(self.n_periods, dtype=np.int64),
            "n_samples": np.asarray(self.n_samples, dtype=np.int64),
            "initial_radius": np.asarray(self.initial_radius, dtype=np.float32),
        }
        if self.fundamentals is not None:
            arrays["fundamentals"] = np.asarray(self.fundamentals, dtype=np.float32)
        if self.lines is not None:
            arrays["lines"] = np.asarray(self.lines, dtype=np.float32)
        np.savez(path, **arrays)
        out = os.fspath(path)
        out = out if out.endswith(".npz") else out + ".npz"
        logger.info(f"Wrote {len(self.ids)} orbits to {out}")
        return out

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "OrbitResults":
        """Reconstruct an :class:`OrbitResults` saved by :meth:`save`.

        Parameters
        ----------
        path : str or os.PathLike
            A ``.npz`` file written by :meth:`save`.

        Returns
        -------
        results : OrbitResults
            The reconstructed results, including frequency data if it was saved.

        Raises
        ------
        ValueError
            If the file is not a lanfear ``OrbitResults`` archive.
        """
        with np.load(path, allow_pickle=False) as npz:
            if "_format" not in npz or str(npz["_format"]) != _RESULTS_FORMAT:
                raise ValueError(
                    f"{os.fspath(path)!r} is not a lanfear OrbitResults file"
                )
            return cls(
                ids=npz["ids"],
                summary=npz["summary"],
                columns=[str(c) for c in npz["columns"]],
                time_unit=float(npz["time_unit"]),
                length_unit=float(npz["length_unit"]),
                n_periods=int(npz["n_periods"]),
                n_samples=int(npz["n_samples"]),
                initial_radius=npz["initial_radius"],
                fundamentals=npz["fundamentals"] if "fundamentals" in npz else None,
                lines=npz["lines"] if "lines" in npz else None,
            )

    def classify(self, **kwargs):
        """Classify these orbits into families.

        Parameters
        ----------
        **kwargs
            Passed through to :func:`lanfear.classify.classify_orbits`.

        Returns
        -------
        classification : lanfear.classify.OrbitClassification
            The per-orbit family labels and diagnostics.
        """
        from .classify import classify_orbits

        return classify_orbits(self, **kwargs)


def _launched_parallel() -> bool:
    """Heuristic test for a multi-rank MPI/SLURM launch.

    Returns
    -------
    parallel : bool
        True if an MPI/SLURM launcher environment variable indicates > 1 rank.
    """
    import os

    for var in ("PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "SLURM_NTASKS"):
        try:
            if int(os.environ.get(var, "1")) > 1:
                return True
        except ValueError:
            pass
    return False


def _resolve_comm(comm):
    """Resolve the requested communicator to an MPI communicator or None.

    "auto" falls back to serial when mpi4py is absent or the MPI runtime cannot
    be initialised -- except when the process was clearly launched in parallel,
    in which case the error is surfaced (silent per-rank serial runs would be a
    trap). Launching via ``mpirun``/``srun`` puts the MPI runtime on the library
    path; a plain ``python`` invocation runs serially.

    Parameters
    ----------
    comm : None, str, or mpi4py.MPI.Comm
        None for serial, ``"auto"`` to use ``COMM_WORLD`` when it has more than
        one rank, or an already-constructed communicator.

    Returns
    -------
    comm : mpi4py.MPI.Comm or None
        The communicator to use, or None for serial execution.

    Raises
    ------
    ValueError
        If ``comm`` is an unrecognised string.
    RuntimeError
        If the process was launched in parallel but MPI cannot be initialised.
    """
    if comm is None:
        return None
    if isinstance(comm, str):
        if comm != "auto":
            raise ValueError(f"unknown comm specifier {comm!r}")
        try:
            from mpi4py import MPI
        except Exception as exc:  # ImportError, or MPI runtime not loadable
            if _launched_parallel():
                raise RuntimeError(
                    "Launched in parallel but mpi4py could not initialise MPI; "
                    "launch via mpirun/srun so the MPI runtime is on the "
                    "library path."
                ) from exc
            return None
        world = MPI.COMM_WORLD
        return world if world.Get_size() > 1 else None
    return comm  # assume an already-constructed communicator


def _split_counts(n: int, size: int) -> list:
    """Balanced per-rank item counts.

    Parameters
    ----------
    n : int
        Total number of items.
    size : int
        Number of ranks.

    Returns
    -------
    counts : list of int
        Item count for each rank (remainder spread over the first ranks).
    """
    base, rem = divmod(n, size)
    return [base + (1 if r < rem else 0) for r in range(size)]


def _scatter_rows(comm, arr, ncol, root):
    """Scatter a 2-D float64 array across ranks by rows.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        The communicator.
    arr : numpy.ndarray
        (N, ncol) array on ``root`` (ignored on other ranks).
    ncol : int
        Number of columns per row.
    root : int
        Rank holding ``arr``.

    Returns
    -------
    local : numpy.ndarray
        This rank's (n_local, ncol) block.
    counts : list of int
        Per-rank row counts.
    """
    from mpi4py import MPI

    rank, size = comm.Get_rank(), comm.Get_size()
    counts = _split_counts(arr.shape[0], size) if rank == root else None
    counts = comm.bcast(counts, root=root)
    local = np.empty((counts[rank], ncol), dtype=np.float64)

    sendbuf = None
    if rank == root:
        send = np.ascontiguousarray(arr, dtype=np.float64)
        sendcounts = [c * ncol for c in counts]
        displs = np.insert(np.cumsum(sendcounts)[:-1], 0, 0)
        sendbuf = [send, sendcounts, displs, MPI.DOUBLE]
    comm.Scatterv(sendbuf, local, root=root)
    return local, counts


def _gather_rows(comm, local, counts, ncol, root):
    """Gather per-rank row blocks into a single array on root.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        The communicator.
    local : numpy.ndarray
        This rank's (n_local, ncol) block.
    counts : list of int
        Per-rank row counts (as returned by :func:`_scatter_rows`).
    ncol : int
        Number of columns per row.
    root : int
        Destination rank.

    Returns
    -------
    out : numpy.ndarray or None
        The (N, ncol) gathered array on ``root``; None on other ranks.
    """
    from mpi4py import MPI

    rank = comm.Get_rank()
    recvbuf = out = None
    if rank == root:
        out = np.empty((sum(counts), ncol), dtype=np.float64)
        recvcounts = [c * ncol for c in counts]
        displs = np.insert(np.cumsum(recvcounts)[:-1], 0, 0)
        recvbuf = [out, recvcounts, displs, MPI.DOUBLE]
    comm.Gatherv(np.ascontiguousarray(local, dtype=np.float64), recvbuf, root=root)
    return out


def _log_integration_result(summary, seconds) -> None:
    """Log an INFO summary of a completed batch, warning on failed orbits.

    Parameters
    ----------
    summary : numpy.ndarray
        (N, len(SUMMARY_COLUMNS)) summary rows (root rank only).
    seconds : float
        Wall-clock time taken by the integration.
    """
    n_tot = len(summary)
    n_ok = int(np.sum(summary[:, _COL_INDEX["status"]] == 0))
    pct_ok = 100.0 * n_ok / max(n_tot, 1)
    logger.info(f"Integrated {n_tot} orbits in {seconds:.1f} s ({pct_ok:.1f}% ok)")
    if n_ok < n_tot:
        logger.warning(
            f"{n_tot - n_ok}/{n_tot} orbits did not integrate cleanly (status != 0)"
        )


def analyse_states(
    scf: "_core.SCFPotential",
    states: Optional[np.ndarray],
    n_periods: int = 50,
    n_samples: int = 8192,
    abs_tol: float = 1e-10,
    rel_tol: float = 1e-9,
    n_lines: int = 4,
    comm="auto",
    root: int = 0,
    progress: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Integrate and frequency-analyse HO-unit states, MPI-distributed.

    Integrates each orbit and extracts the leading ``n_lines`` spectral lines
    per axis (the frequency data that drives classification).

    Parameters
    ----------
    scf : lanfear._core.SCFPotential or lanfear._core.DiscPotential
        The C++ potential; need only be valid on ``root`` (it is broadcast).
    states : numpy.ndarray or None
        (N, 6) initial states in HO units; need only be valid on ``root``.
    n_periods : int, optional
        Number of orbital periods to integrate.
    n_samples : int, optional
        Number of uniform samples per orbit.
    abs_tol : float, optional
        Absolute tolerance of the adaptive integrator.
    rel_tol : float, optional
        Relative tolerance of the adaptive integrator.
    n_lines : int, optional
        Number of leading spectral lines extracted per axis.
    comm : None, str, or mpi4py.MPI.Comm, optional
        Communicator selector (see :func:`_resolve_comm`).
    root : int, optional
        Rank holding the inputs and receiving the result.
    progress : bool, optional
        If True (default), the C++ core prints ``"<X>% of particles integrated"``
        every 10% of orbits. Under MPI only the ``root`` rank reports (on its own
        share of the orbits) to avoid interleaved output from every rank.

    Returns
    -------
    summary : numpy.ndarray or None
        (N, len(SUMMARY_COLUMNS)) dynamics summary on ``root``.
    fundamentals : numpy.ndarray or None
        (N, 3) fundamental frequencies on ``root``.
    lines : numpy.ndarray or None
        (N, 3, n_lines, 2) leading spectral lines on ``root``. All three are
        None on non-root ranks.
    """
    comm = _resolve_comm(comm)
    ncol = len(SUMMARY_COLUMNS)
    line_cols = 3 * n_lines * 2

    def _reshape_lines(flat):
        """Reshape a flattened lines block back to (N, 3, n_lines, 2).

        Parameters
        ----------
        flat : numpy.ndarray or None
            (N, 3*n_lines*2) flattened lines, or None.

        Returns
        -------
        lines : numpy.ndarray or None
            The reshaped (N, 3, n_lines, 2) array, or None if ``flat`` is None.
        """
        return None if flat is None else flat.reshape(-1, 3, n_lines, 2)

    if comm is None:  # serial
        if states is None:
            raise ValueError("states must be provided for serial analysis")
        summ, fund, lines = scf.analyse_batch(
            np.ascontiguousarray(states, dtype=np.float64),
            n_periods,
            n_samples,
            abs_tol,
            rel_tol,
            n_lines,
            progress,
        )
        return summ, fund, lines

    rank = comm.Get_rank()
    scf = comm.bcast(scf if rank == root else None, root=root)
    if rank == root and states is None:
        raise ValueError("states must be provided on the root rank")

    local_states, counts = _scatter_rows(
        comm, states if rank == root else np.empty((0, 6)), 6, root
    )
    l_summ, l_fund, l_lines = scf.analyse_batch(
        local_states,
        n_periods,
        n_samples,
        abs_tol,
        rel_tol,
        n_lines,
        progress and rank == root,
    )

    summary = _gather_rows(comm, l_summ, counts, ncol, root)
    fundamentals = _gather_rows(comm, l_fund, counts, 3, root)
    lines_flat = _gather_rows(
        comm, l_lines.reshape(len(l_lines), line_cols), counts, line_cols, root
    )
    return summary, fundamentals, _reshape_lines(lines_flat)


def analyse_family(
    potential: Potential,
    particles: Optional[ParticleSystem],
    family: Union[str, Sequence[str]] = "STAR",
    n_periods: int = 50,
    n_samples: int = 8192,
    abs_tol: float = 1e-10,
    rel_tol: float = 1e-9,
    n_lines: int = 4,
    comm="auto",
    root: int = 0,
    progress: bool = True,
) -> Optional[OrbitResults]:
    """Integrate and frequency-analyse every particle of the given family.

    Selects the family from ``particles``, integrates each orbit in
    ``potential``, and returns an :class:`OrbitResults` carrying the per-orbit
    summary plus ``fundamentals`` (N, 3) and ``lines`` (N, 3, n_lines, 2) for
    resonance-based classification.

    Parameters
    ----------
    potential : Potential or DiscPotential
        The analytical potential; need only be valid on ``root``.
    particles : ParticleSystem or None
        The particle system; need only be valid on ``root``.
    family : str or sequence of str, optional
        Species label(s) to integrate (default ``"STAR"``).
    n_periods : int, optional
        Number of orbital periods to integrate.
    n_samples : int, optional
        Number of uniform samples per orbit.
    abs_tol : float, optional
        Absolute tolerance of the adaptive integrator.
    rel_tol : float, optional
        Relative tolerance of the adaptive integrator.
    n_lines : int, optional
        Number of leading spectral lines extracted per axis.
    comm : None, str, or mpi4py.MPI.Comm, optional
        Communicator selector (see :func:`_resolve_comm`).
    root : int, optional
        Rank holding the inputs and receiving the result.
    progress : bool, optional
        If True (default), the C++ core prints ``"<X>% of particles integrated"``
        every 10% of orbits (root rank only under MPI).

    Returns
    -------
    results : OrbitResults or None
        Per-orbit results (with frequency data) on ``root``; None on other ranks.

    Raises
    ------
    ValueError
        On ``root`` if ``particles`` is None or no particles match ``family``.
    """
    resolved = _resolve_comm(comm)
    rank = resolved.Get_rank() if resolved is not None else 0
    size = resolved.Get_size() if resolved is not None else 1

    states = ids = None
    if rank == root:
        if particles is None:
            raise ValueError("particles must be provided on the root rank")
        labels = (family,) if isinstance(family, str) else tuple(family)
        sub = particles.select(particles.species_mask(*labels))
        if sub.n_particles == 0:
            raise ValueError(f"no particles matched family {labels}")
        states = potential.to_ho_state(sub.pos, sub.vel)
        ids = sub.ids
        workers = f"{size} MPI ranks" if size > 1 else "1 process (serial)"
        logger.info(
            f"Integrating + frequency-analysing {sub.n_particles} orbits "
            f"(family={labels}) for {n_periods} periods on {workers}"
        )

    t0 = time.perf_counter()
    summary, fundamentals, lines = analyse_states(
        potential.core if rank == root else None,
        states,
        n_periods=n_periods,
        n_samples=n_samples,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        n_lines=n_lines,
        comm=resolved,
        root=root,
        progress=progress,
    )

    if rank != root:
        return None
    _log_integration_result(summary, time.perf_counter() - t0)
    return OrbitResults(
        ids=np.asarray(ids),
        summary=summary,
        columns=SUMMARY_COLUMNS,
        time_unit=potential.time_unit,
        n_periods=n_periods,
        n_samples=n_samples,
        fundamentals=fundamentals,
        lines=lines,
        length_unit=potential.scale_radius,
        initial_radius=np.linalg.norm(states[:, :3], axis=1),
    )


class ParticleTrajectory:
    """The phase-space trajectory of a single particle.

    Keeping the full trajectory of every particle in an :class:`OrbitResults`
    batch "just in case" would be wasteful; in practice only a handful of
    orbits are ever inspected in this much detail. This class instead
    re-integrates one particle on demand (via
    ``potential.core.integrate_orbit``, the single-orbit counterpart of the
    batch integrator) and keeps only that trajectory, in physical units.

    Parameters
    ----------
    particle_id : int or None
        Identifier of the particle this trajectory belongs to.
    pos : numpy.ndarray
        (n_samples, 3) positions in physical length units, uniformly sampled
        in time.
    time : numpy.ndarray
        (n_samples,) physical times at which ``pos`` was sampled.
    status : int
        Integrator status (0 ok, 1 period estimate failed, 2 NaN
        encountered); ``pos``/``time`` may be short or empty when non-zero.
    """

    def __init__(
        self,
        particle_id: Optional[int],
        pos: np.ndarray,
        time: np.ndarray,
        status: int,
    ) -> None:
        self.particle_id = particle_id
        self.pos = pos
        self.time = time
        self.status = status

    @classmethod
    def integrate(
        cls,
        potential: Potential,
        pos_phys,
        vel_phys,
        particle_id: Optional[int] = None,
        n_periods: int = 50,
        n_samples: int = 2000,
        abs_tol: float = 1e-10,
        rel_tol: float = 1e-9,
    ) -> "ParticleTrajectory":
        """Integrate one particle and keep its trajectory.

        Parameters
        ----------
        potential : Potential
            The analytical potential to integrate in.
        pos_phys : array-like of float
            (3,) particle position in physical length units.
        vel_phys : array-like of float
            (3,) particle velocity in physical velocity units.
        particle_id : int, optional
            Identifier to attach to the trajectory (for labelling plots).
        n_periods : int, optional
            Number of orbital periods to integrate.
        n_samples : int, optional
            Number of uniformly time-spaced trajectory samples to keep.
        abs_tol : float, optional
            Absolute tolerance of the adaptive integrator.
        rel_tol : float, optional
            Relative tolerance of the adaptive integrator.

        Returns
        -------
        trajectory : ParticleTrajectory
            The integrated trajectory, in physical units.
        """
        state = potential.to_ho_state(pos_phys, vel_phys)[0]
        summary, traj_ho = potential.core.integrate_orbit(
            state,
            n_periods=n_periods,
            n_samples=n_samples,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            return_trajectory=True,
        )
        status = int(summary[_COL_INDEX["status"]])
        if status != 0:
            logger.warning(
                f"Particle {particle_id} did not integrate cleanly "
                f"(status={status})"
            )

        # Samples are uniformly spaced at dt_out = t_total / (n_samples - 1);
        # reconstruct time from that spacing (rather than assuming n_samples
        # rows) so a trajectory truncated by a mid-integration NaN still gets
        # correctly timed samples.
        t_total_ho = summary[_COL_INDEX["t_total"]]
        dt_out_ho = t_total_ho / (n_samples - 1)
        time = dt_out_ho * np.arange(len(traj_ho)) * potential.time_unit
        pos = traj_ho[:, :3] * potential.scale_radius
        return cls(particle_id=particle_id, pos=pos, time=time, status=status)

    @classmethod
    def from_particles(
        cls,
        potential: Potential,
        particles: ParticleSystem,
        particle_id: int,
        **kwargs,
    ) -> "ParticleTrajectory":
        """Integrate the trajectory of one particle picked out of a system.

        Parameters
        ----------
        potential : Potential
            The analytical potential to integrate in.
        particles : ParticleSystem
            The system to look ``particle_id`` up in.
        particle_id : int
            Particle identifier (matched against ``particles.ids``).
        **kwargs
            Passed through to :meth:`integrate`.

        Returns
        -------
        trajectory : ParticleTrajectory
            The integrated trajectory.

        Raises
        ------
        ValueError
            If no particle in ``particles`` has the given ``particle_id``.
        """
        matches = np.flatnonzero(particles.ids == particle_id)
        if matches.size == 0:
            raise ValueError(f"no particle with id={particle_id!r} in particles")
        index = int(matches[0])
        return cls.integrate(
            potential,
            particles.pos[index],
            particles.vel[index],
            particle_id=particle_id,
            **kwargs,
        )

    def plot(
        self,
        axes=None,
        colourmap: str = "BuPu",
        white_fraction: float = 0.15,
        linewidth: float = 1.0,
    ):
        """Plot the trajectory as x-y, x-z and y-z projections.

        Each projection is drawn as a single line coloured along its length by
        the trajectory time, using a :class:`~matplotlib.collections.LineCollection`
        so a many-thousand-sample trajectory is one draw call per axis rather
        than one per segment.

        Parameters
        ----------
        axes : sequence of matplotlib.axes.Axes, optional
            Three axes (x-y, x-z, y-z) to draw into. A new 1x3 figure is
            created if omitted. Each panel is drawn square, sharing one
            symmetric, origin-centred scale (sized to the trajectory's
            largest coordinate) so the three projections are directly
            comparable.
        colourmap : str, optional
            Name of a sequential matplotlib colourmap to trace time along the
            trajectory (default ``"BuPu"``).
        white_fraction : float, optional
            Fraction of the colourmap's low (near-white) end to discard, so
            the earliest-time segments stay visible against a white
            background.
        linewidth : float, optional
            Width of the trajectory line.

        Returns
        -------
        axes : numpy.ndarray of matplotlib.axes.Axes
            The (x-y, x-z, y-z) axes drawn on.
        """
        if len(self.pos) < 2:
            raise ValueError("trajectory has fewer than 2 samples to plot")

        if axes is None:
            # constrained layout keeps tick/axis labels from bleeding into
            # the neighbouring panel and trims the dead space around the
            # figure -- and, unlike tight_layout(), it stays valid if this
            # method is called again on the same figure (e.g. to redraw).
            _, axes = plt.subplots(1, 3, figsize=(10, 3.3), layout="constrained")
        axes = np.asarray(axes)

        base_colours = plt.get_cmap(colourmap)
        cmap = LinearSegmentedColormap.from_list(
            f"{colourmap}_no_white",
            base_colours(np.linspace(white_fraction, 1.0, 256)),
        )
        normaliser = Normalize(vmin=self.time[0], vmax=self.time[-1])
        segment_time = 0.5 * (self.time[:-1] + self.time[1:])

        # A shared, symmetric extent (rather than each axes' own autoscale)
        # so every projection uses the same physical scale; combined with
        # set_aspect("equal", "box") below, that makes each panel square.
        half_extent = 1.05 * np.max(np.abs(self.pos))

        projections = (("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2))
        collection = None
        for ax, (xlabel, ylabel, i, j) in zip(axes, projections):
            coords = self.pos[:, (i, j)]
            segments = np.stack([coords[:-1], coords[1:]], axis=1)
            collection = LineCollection(
                segments, cmap=cmap, norm=normaliser, linewidth=linewidth
            )
            collection.set_array(segment_time)
            ax.add_collection(collection)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xlim(-half_extent, half_extent)
            ax.set_ylim(-half_extent, half_extent)
            ax.set_aspect("equal", adjustable="box")

        # Colourbar keyed to axes[-1] alone (not the full three-axes group):
        # under constrained layout that sizes it to that axes' actual
        # rendered height -- which set_aspect("equal", "box") may have
        # shrunk below its nominal slot -- so it always matches, and
        # constrained layout (unlike an axes_grid1 divider) reserves room
        # for its label instead of clipping it at the figure edge.
        fig = axes[0].figure
        fig.colorbar(collection, ax=axes[-1], label="time")
        fig.suptitle(f"ID: {self.particle_id}")
        return axes
