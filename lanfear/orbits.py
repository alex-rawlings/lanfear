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

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from . import _core
from .particle_system import ParticleSystem
from .potential import Potential

SUMMARY_COLUMNS = list(_core.summary_columns())
_COL_INDEX = {name: i for i, name in enumerate(SUMMARY_COLUMNS)}


@dataclass
class OrbitResults:
    """Per-orbit integration/analysis results (populated on the root rank only).

    ``fundamentals`` and ``lines`` are present only for results produced by
    :func:`analyse_family` / :func:`analyse_states` (frequency analysis). All
    frequencies are signed angular frequencies in HO units (rad / HO time); a
    negative sign encodes the sense of circulation. Multiply by
    ``1 / time_unit`` for physical angular frequency.
    """

    ids: np.ndarray  # (N,) particle IDs
    summary: np.ndarray  # (N, len(SUMMARY_COLUMNS))
    columns: Sequence[str]  # column names for `summary`
    time_unit: float  # HO time -> physical time factor
    n_periods: int
    n_samples: int
    fundamentals: Optional[np.ndarray] = None  # (N, 3) leading freq per axis
    lines: Optional[np.ndarray] = None  # (N, 3, n_lines, 2) freq, amp

    def column(self, name: str) -> np.ndarray:
        """Return the named summary column (see :data:`SUMMARY_COLUMNS`)."""
        return self.summary[:, _COL_INDEX[name]]

    @property
    def period_physical(self) -> np.ndarray:
        """Estimated orbital period of each particle in physical time units."""
        return self.column("period") * self.time_unit

    @property
    def ok(self) -> np.ndarray:
        """Boolean mask of orbits that integrated successfully (status == 0)."""
        return self.column("status") == 0

    @property
    def frequency_ratios(self) -> np.ndarray:
        """(N, 2) ratios |w_x|/|w_z|, |w_y|/|w_z| of the fundamentals.

        The building block of resonance-based classification. NaN where a
        denominator vanishes (e.g. an axis with no oscillation).
        """
        if self.fundamentals is None:
            raise ValueError("no frequency data; use analyse_family/analyse_states")
        w = np.abs(self.fundamentals)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.stack([w[:, 0] / w[:, 2], w[:, 1] / w[:, 2]], axis=1)

    def to_dict(self) -> dict:
        d = {name: self.summary[:, i] for i, name in enumerate(self.columns)}
        d["id"] = self.ids
        if self.fundamentals is not None:
            d["freq_x"] = self.fundamentals[:, 0]
            d["freq_y"] = self.fundamentals[:, 1]
            d["freq_z"] = self.fundamentals[:, 2]
        return d

    def classify(self, **kwargs):
        """Classify these orbits into families (see :func:`classify_orbits`)."""
        from .classify import classify_orbits

        return classify_orbits(self, **kwargs)


def _launched_parallel() -> bool:
    """Heuristic: were we started under a multi-rank MPI/SLURM launcher?"""
    import os

    for var in ("PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "SLURM_NTASKS"):
        try:
            if int(os.environ.get(var, "1")) > 1:
                return True
        except ValueError:
            pass
    return False


def _resolve_comm(comm):
    """Return an MPI communicator, or None for serial execution.

    ``comm`` may be None (serial), the string "auto" (use COMM_WORLD if it has
    more than one rank and MPI is usable), or an explicit communicator.

    "auto" falls back to serial when mpi4py is absent or the MPI runtime cannot
    be initialised -- except when the process was clearly launched in parallel,
    in which case the error is surfaced (silent per-rank serial runs would be a
    trap). Launching via ``mpirun``/``srun`` puts the MPI runtime on the library
    path; a plain ``python`` invocation runs serially.
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
    """Balanced particle counts per rank (remainder spread over first ranks)."""
    base, rem = divmod(n, size)
    return [base + (1 if r < rem else 0) for r in range(size)]


def _scatter_rows(comm, arr, ncol, root):
    """Scatter a (N, ncol) float64 array by rows; returns this rank's block."""
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
    """Gather (n_local, ncol) blocks back into a (N, ncol) array on root."""
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


def integrate_states(
    scf: "_core.SCFPotential",
    states: Optional[np.ndarray],
    n_periods: int = 50,
    n_samples: int = 8192,
    abs_tol: float = 1e-10,
    rel_tol: float = 1e-9,
    comm="auto",
    root: int = 0,
) -> Optional[np.ndarray]:
    """Integrate a set of HO-unit states, MPI-distributed, returning summaries.

    ``scf`` need only be valid on ``root``; it is broadcast to all ranks.
    ``states`` (N,6) need only be valid on ``root``. Returns the (N, ncols)
    summary array on ``root`` and None elsewhere. With no usable communicator it
    runs serially.
    """
    comm = _resolve_comm(comm)
    ncol = len(SUMMARY_COLUMNS)

    if comm is None:  # serial
        if states is None:
            raise ValueError("states must be provided for serial integration")
        return scf.integrate_batch(
            np.ascontiguousarray(states, dtype=np.float64),
            n_periods,
            n_samples,
            abs_tol,
            rel_tol,
        )

    rank = comm.Get_rank()
    # Broadcast the (compact) potential so only root builds the coefficients.
    scf = comm.bcast(scf if rank == root else None, root=root)
    if rank == root and states is None:
        raise ValueError("states must be provided on the root rank")

    local_states, counts = _scatter_rows(
        comm, states if rank == root else np.empty((0, 6)), 6, root
    )
    local_summary = scf.integrate_batch(
        local_states, n_periods, n_samples, abs_tol, rel_tol
    )
    return _gather_rows(comm, local_summary, counts, ncol, root)


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
):
    """Integrate + frequency-analyse HO-unit states, MPI-distributed.

    Like :func:`integrate_states` but also extracts the leading ``n_lines``
    spectral lines per axis. Returns ``(summary (N,ncols), fundamentals (N,3),
    lines (N,3,n_lines,2))`` on ``root`` (a tuple of Nones elsewhere).
    """
    comm = _resolve_comm(comm)
    ncol = len(SUMMARY_COLUMNS)
    line_cols = 3 * n_lines * 2

    def _reshape_lines(flat):
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
        local_states, n_periods, n_samples, abs_tol, rel_tol, n_lines
    )

    summary = _gather_rows(comm, l_summ, counts, ncol, root)
    fundamentals = _gather_rows(comm, l_fund, counts, 3, root)
    lines_flat = _gather_rows(
        comm, l_lines.reshape(len(l_lines), line_cols), counts, line_cols, root
    )
    return summary, fundamentals, _reshape_lines(lines_flat)


def integrate_family(
    potential: Potential,
    particles: Optional[ParticleSystem],
    family: Union[str, Sequence[str]] = "STAR",
    n_periods: int = 50,
    n_samples: int = 8192,
    abs_tol: float = 1e-10,
    rel_tol: float = 1e-9,
    comm="auto",
    root: int = 0,
) -> Optional[OrbitResults]:
    """Integrate every particle of the given family in ``potential``.

    ``potential`` and ``particles`` need only be valid on the ``root`` rank.
    Returns an :class:`OrbitResults` on ``root`` and None on other ranks.
    """
    resolved = _resolve_comm(comm)
    rank = resolved.Get_rank() if resolved is not None else 0

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

    summary = integrate_states(
        potential.core if rank == root else None,
        states,
        n_periods=n_periods,
        n_samples=n_samples,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        comm=resolved,
        root=root,
    )

    if rank != root:
        return None
    return OrbitResults(
        ids=np.asarray(ids),
        summary=summary,
        columns=SUMMARY_COLUMNS,
        time_unit=potential.time_unit,
        n_periods=n_periods,
        n_samples=n_samples,
    )


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
) -> Optional[OrbitResults]:
    """Integrate + frequency-analyse every particle of the given family.

    Like :func:`integrate_family`, but the returned :class:`OrbitResults` also
    carries ``fundamentals`` (N,3) and ``lines`` (N,3,n_lines,2) for
    resonance-based classification. Root-only; None on other ranks.
    """
    resolved = _resolve_comm(comm)
    rank = resolved.Get_rank() if resolved is not None else 0

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
    )

    if rank != root:
        return None
    return OrbitResults(
        ids=np.asarray(ids),
        summary=summary,
        columns=SUMMARY_COLUMNS,
        time_unit=potential.time_unit,
        n_periods=n_periods,
        n_samples=n_samples,
        fundamentals=fundamentals,
        lines=lines,
    )
