"""Sensitivity of the orbit classification to the pattern speed.

A rotating figure's pattern speed is estimated from a single snapshot with an
uncertainty (:meth:`lanfear.ParticleSystem.estimate_pattern_speed`), and the
classification of a rotating figure depends on it. :func:`pattern_speed_sensitivity`
re-integrates a subsample of orbits at ``Omega_p - n sigma``, ``Omega_p`` and
``Omega_p + n sigma`` (about the rotation axis) and reports which orbits change
label. When no rotation was detected (the pattern speed was set to zero because
it was below the significance cut), the test rates are ``+-n sigma`` about the
short axis z, so the test asks whether an undetected rotation would matter.

Integration is expensive, so the test is opt-in and runs on a subsample; it
uses :func:`lanfear.analyse_family` and so runs under MPI like it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ._logging import get_logger
from .classify import OrbitClassification
from .orbits import _resolve_comm, analyse_family

logger = get_logger(__name__)

# Label arrays compared between the three runs.
_COMPARED = ("labels", "shape", "sense", "pattern_resonance", "regularity")


@dataclass
class PatternSpeedSensitivity:
    """Result of :func:`pattern_speed_sensitivity`.

    Parameters
    ----------
    ids : numpy.ndarray
        (N,) particle IDs of the tested orbits.
    pattern_speeds : numpy.ndarray
        (3, 3) pattern-speed vectors of the three runs (physical units):
        ``Omega - delta``, ``Omega``, ``Omega + delta``.
    axis : int
        Axis the pattern speed was varied about (0=x, 1=y, 2=z).
    delta : float
        Size of the variation, ``n_sigma`` times the uncertainty about
        ``axis`` (physical units).
    classifications : tuple of OrbitClassification
        The three classifications, in the order of :attr:`pattern_speeds`,
        aligned on :attr:`ids`.
    flips : dict
        Label name (``"labels"``, ``"shape"``, ``"sense"``,
        ``"pattern_resonance"``, ``"regularity"``) -> (N,) bool, True where
        either varied run disagrees with the nominal one.
    runaway_changed : numpy.ndarray
        (N,) True where the orbit runs away in some but not all runs.
    """

    ids: np.ndarray
    pattern_speeds: np.ndarray
    axis: int
    delta: float
    classifications: Tuple[
        OrbitClassification, OrbitClassification, OrbitClassification
    ]
    flips: Dict[str, np.ndarray]
    runaway_changed: np.ndarray

    @property
    def flipped(self) -> np.ndarray:
        """Orbits whose family label or runaway status changes.

        Returns
        -------
        mask : numpy.ndarray
            (N,) True where the family label (:attr:`OrbitClassification.labels`)
            differs between runs, or the orbit runs away in only some of them.
        """
        return self.flips["labels"] | self.runaway_changed

    @property
    def flip_fractions(self) -> Dict[str, float]:
        """Fraction of the tested orbits that flip, per label kind.

        Returns
        -------
        fractions : dict
            Label name -> fraction of orbits flipping, plus ``"runaway"`` and
            ``"any"`` (:attr:`flipped`).
        """
        n = max(len(self.ids), 1)
        fractions = {name: float(mask.sum()) / n for name, mask in self.flips.items()}
        fractions["runaway"] = float(self.runaway_changed.sum()) / n
        fractions["any"] = float(self.flipped.sum()) / n
        return fractions


def _variations(potential, n_sigma: float):
    """The three pattern speeds to test and the axis/size of the variation.

    Parameters
    ----------
    potential : potential object
        Carries ``pattern_speed`` and ``pattern_speed_sigma``.
    n_sigma : float
        Size of the variation in units of the uncertainty.

    Returns
    -------
    speeds : numpy.ndarray
        (3, 3) pattern speeds: minus, nominal, plus.
    axis : int
        Axis varied about.
    delta : float
        Size of the variation.

    Raises
    ------
    ValueError
        If the potential has no pattern-speed uncertainty, or it is not a
        positive finite number about the varied axis.
    """
    sigma = potential.pattern_speed_sigma
    if sigma is None:
        raise ValueError(
            "The potential has no pattern-speed uncertainty (pattern_speed_sigma "
            "is None: the pattern speed was given rather than estimated)"
        )
    omega = np.asarray(potential.pattern_speed, dtype=np.float64)
    axis = int(np.argmax(np.abs(omega))) if np.any(omega) else 2
    delta = n_sigma * float(sigma[axis])
    if not (np.isfinite(delta) and delta > 0):
        raise ValueError(
            f"The pattern-speed uncertainty about axis {'xyz'[axis]} is "
            f"{sigma[axis]!r}; cannot vary the pattern speed by n_sigma={n_sigma:g} of it"
        )
    step = np.zeros(3)
    step[axis] = delta
    return np.stack([omega - step, omega, omega + step]), axis, delta


def pattern_speed_sensitivity(
    potential,
    particles,
    family: Union[str, Sequence[str]] = "STAR",
    n_sigma: float = 1.0,
    subsample: Optional[int] = 1000,
    seed: int = 0,
    n_periods: int = 50,
    n_samples: int = 2048,
    comm="auto",
    root: int = 0,
    classify_kwargs: Optional[dict] = None,
    **analyse_kwargs,
) -> Optional[PatternSpeedSensitivity]:
    """How many orbit labels change when the pattern speed changes by ``n_sigma``.

    Integrates a random subsample of the family three times, at
    ``Omega - delta``, ``Omega`` and ``Omega + delta``, with ``delta = n_sigma``
    times the pattern-speed uncertainty about the rotation axis (the axis of
    the largest component; z if the pattern speed is zero, i.e. the rotation
    was below the estimator's significance cut), and classifies each run.
    The potential's pattern speed is restored afterwards.

    Collective under MPI: every rank must call it; ``potential`` and
    ``particles`` need only be valid on ``root``.

    Parameters
    ----------
    potential : Potential, DiscPotential or MultiComponentPotential
        The potential, with ``pattern_speed`` and ``pattern_speed_sigma`` (set
        when the pattern speed was estimated by
        :meth:`lanfear.ParticleSystem.prepare`).
    particles : lanfear.ParticleSystem
        The prepared particle system.
    family : str or sequence of str, optional
        Species to sample (see :func:`lanfear.analyse_family`).
    n_sigma : float, optional
        Size of the variation in units of the uncertainty (default 1).
    subsample : int or None, optional
        Number of orbits to test (default 1000); None tests the whole family.
    seed : int, optional
        Seed of the subsample draw.
    n_periods, n_samples : int, optional
        Integration settings (see :func:`lanfear.analyse_family`).
    comm : None, str or mpi4py.MPI.Comm, optional
        Communicator selector (see :func:`lanfear.analyse_family`).
    root : int, optional
        Rank holding the inputs and receiving the result.
    classify_kwargs : dict, optional
        Passed to :meth:`lanfear.OrbitResults.classify` for every run
        (``drop_runaways`` is always False, so the runs stay aligned; runaway
        changes are reported separately).
    **analyse_kwargs
        Passed to :func:`lanfear.analyse_family`.

    Returns
    -------
    sensitivity : PatternSpeedSensitivity or None
        The result on ``root``; None on other ranks.

    Raises
    ------
    ValueError
        On every rank, if the potential has no usable pattern-speed
        uncertainty.
    """
    comm = _resolve_comm(comm)
    rank = 0 if comm is None else comm.Get_rank()
    classify_kwargs = dict(classify_kwargs or {})
    classify_kwargs["drop_runaways"] = False

    # Decide on root, then make every rank agree before any collective call.
    setup = error = None
    if rank == root:
        try:
            speeds, axis, delta = _variations(potential, n_sigma)
            selected = particles.select(particles.species_mask(*np.atleast_1d(family)))
            if subsample is not None:
                selected = selected.random_subset(
                    subsample, rng=np.random.default_rng(seed)
                )
            setup = (speeds, axis, delta, selected)
        except ValueError as exc:
            error = exc
    failed = error is not None
    if comm is not None:
        failed = comm.bcast(failed, root=root)
    if failed:
        if rank == root:
            raise error
        raise ValueError("pattern_speed_sensitivity failed on the root rank")

    speeds, axis, delta, selected = setup if rank == root else (None,) * 4
    original = sigma = None
    if rank == root:
        original = potential.pattern_speed
        sigma = potential.pattern_speed_sigma
        logger.info(
            f"Pattern-speed sensitivity: {selected.n_particles} orbits at "
            f"Omega_{'xyz'[axis]} {speeds[1][axis]:.4g} -+ {delta:.4g} "
            f"({n_sigma:g} sigma)"
        )
    results = []
    try:
        for k in range(3):
            if rank == root:
                potential.pattern_speed = speeds[k]
            results.append(
                analyse_family(
                    potential if rank == root else None,
                    selected,
                    family=family,
                    n_periods=n_periods,
                    n_samples=n_samples,
                    comm=comm,
                    root=root,
                    **analyse_kwargs,
                )
            )
    finally:
        if rank == root:
            potential.pattern_speed = original  # (clears the uncertainty...)
            potential.pattern_speed_sigma = sigma  # ...so restore it too
    if rank != root:
        return None

    classifications = tuple(res.classify(**classify_kwargs) for res in results)
    nominal = classifications[1]
    flips = {}
    for name in _COMPARED:
        reference = getattr(nominal, name)
        flips[name] = np.zeros(len(reference), dtype=bool)
        for k in (0, 2):
            flips[name] |= getattr(classifications[k], name) != reference
    runaway = np.stack([res.runaway for res in results])
    sensitivity = PatternSpeedSensitivity(
        ids=np.asarray(results[1].ids),
        pattern_speeds=speeds,
        axis=axis,
        delta=delta,
        classifications=classifications,
        flips=flips,
        runaway_changed=runaway.any(axis=0) & ~runaway.all(axis=0),
    )
    summary = ", ".join(f"{k} {v:.1%}" for k, v in sensitivity.flip_fractions.items())
    logger.info(f"Pattern-speed sensitivity: labels flipping: {summary}")
    return sensitivity
