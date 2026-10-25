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

Figure rotation: a potential with a non-zero ``pattern_speed`` (see
:meth:`lanfear.ParticleSystem.prepare`) is a figure rotating rigidly at that
angular velocity. Orbits are always integrated in the inertial frame, in the
rotating potential ``Phi(R(t)^T x)``, so the static case (zero pattern speed)
uses exactly the same machinery. Every per-orbit quantity (summary columns,
frequencies, trajectories) is measured in the co-rotating frame of the figure,
where the potential is static and a regular orbit is a steady 3-torus: its
positions are the body-frame ones and its velocities ``dx_b/dt``. The
``energy*`` columns hold the Jacobi integral ``E_J = E - Omega . L``, which is
conserved in a rotating potential and reduces to the energy ``E`` when static.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

from . import _core
from ._logging import get_logger
from .disc_potential import DiscPotential
from .multi_component_potential import MultiComponentPotential
from .particle_system import ParticleSystem
from .potential import Potential

# Any potential-like object usable by analyse_family/analyse_states/
# ParticleTrajectory: exposes .core, .to_ho_state, .scale_radius, .time_unit,
# .n_max, .l_max (see lanfear._potential_base._PotentialBase).
AnyPotential = Union[Potential, DiscPotential, MultiComponentPotential]

logger = get_logger(__name__)

SUMMARY_COLUMNS = list(_core.summary_columns())
_COL_INDEX = {name: i for i, name in enumerate(SUMMARY_COLUMNS)}

# Default cap on the power-of-two factor by which a rotating figure's orbits are
# integrated for longer, to span n_periods body-frame periods (mirrors the C++
# kMaxBodyPeriodFactor; see OrbitResults.body_periods_integrated).
MAX_BODY_PERIOD_FACTOR = 8

# Seed of the fixed permutation that spreads orbits evenly across MPI ranks in
# analyse_states (any fixed value; the results do not depend on it).
_SCATTER_SEED = 20261001

# Tag written into OrbitResults.save() archives so load() can validate them.
_RESULTS_FORMAT = "lanfear.OrbitResults"


def _diffusion_rate(diffusion, lines, amp_frac) -> np.ndarray:
    """Reduce per-axis diffusion rates to one value per orbit.

    Takes the maximum over the axes that actually oscillate: an axis whose
    leading-line amplitude is below ``amp_frac`` of the strongest axis is ignored
    (its frequency is noise-dominated).

    Parameters
    ----------
    diffusion : numpy.ndarray
        (N, 3) per-axis rates (NaN where not measurable).
    lines : numpy.ndarray
        (N, 3, n_lines, 2) spectral lines, used for the axis amplitudes.
    amp_frac : float
        Relative amplitude below which an axis is treated as inactive.

    Returns
    -------
    rate : numpy.ndarray
        (N,) rate; NaN where no active axis has a valid value.
    """
    d = np.asarray(diffusion, dtype=np.float64).copy()
    amp = np.abs(np.asarray(lines[:, :, 0, 1], dtype=np.float64))
    d[amp < amp_frac * amp.max(axis=1, keepdims=True)] = np.nan
    with np.errstate(all="ignore"):
        rate = np.nanmax(np.where(np.isfinite(d), d, -np.inf), axis=1)
    rate[~np.isfinite(rate)] = np.nan
    return rate


@dataclass
class OrbitResults:
    """Per-orbit integration/analysis results (populated on the root rank only).

    ``fundamentals``, ``lines`` and ``diffusion`` carry the frequency data
    classification needs; :func:`analyse_family` / :func:`analyse_states` always
    populate them, and they are required. All frequencies are signed angular frequencies in HO
    units (rad / HO time); a
    negative sign encodes the sense of circulation. Multiply by ``1 / time_unit``
    for physical angular frequency.

    When the potential rotates (a non-zero :attr:`pattern_speed`), the summary
    columns and frequencies are measured in the co-rotating frame of the
    figure, and the ``energy0``/``energy_mean``/``energy_drift`` columns refer
    to the Jacobi integral ``E - Omega . L`` (see :mod:`lanfear.orbits`).

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
    fundamentals : numpy.ndarray
        (N, 3) signed fundamental frequency per axis (HO units).
    lines : numpy.ndarray
        (N, 3, n_lines, 2) leading (freq, amp) spectral lines.
    diffusion : numpy.ndarray
        (N, 3) Laskar frequency-diffusion rate per axis, ``|w2 - w1| / |w1|``
        with ``w1``/``w2`` the leading frequencies of the first/second half of
        the integration (NaN where it could not be measured).
        See :attr:`diffusion_rate`.
    planar_lines : numpy.ndarray
        (N, 3, n_lines, 2) leading signed (freq, amp) lines of the planar
        position signals about each axis, ``y + i z`` (about x), ``z + i x``
        (about y) and ``x + i y`` (about z), DC removed. A loop circulating
        about an axis in the sense of positive ``L`` there has a positive
        leading line; a box has comparable lines at ``+w`` and ``-w``. For the
        plane perpendicular to the pattern speed, the inertial-frame spectrum
        is this one shifted by exactly ``+|Omega_p|``.
    n_periods_used : numpy.ndarray, optional
        (N,) periods each orbit was actually integrated for. Equals
        ``n_periods`` except for orbits extended because their frequency drift was
        high (see :func:`analyse_family`). Only set when extension was enabled.
    diffusion_previous : numpy.ndarray, optional
        (N,) :meth:`diffusion_rate` of each extended orbit at the integration
        length *before* its last extension (NaN for orbits never extended).
        Comparing it with the current rate tells drift that is falling with the
        window (a regular orbit) from drift that is not (chaos); see
        :func:`lanfear.classify.classify_orbits`.
    length_unit : float
        HO-length -> physical-length conversion factor (the scale radius). A
        radius in HO units becomes physical when multiplied by this.
    initial_radius : numpy.ndarray
        (N,) instantaneous radius of each orbit at integration start, i.e. the
        snapshot radius from the galaxy centre, in HO units.
    source_file : str, optional
        Path of the particle-data file the orbits were drawn from.
    n_max : int, optional
        Radial truncation order used to build the potential's SCF expansion.
    l_max : int, optional
        Angular truncation order used to build the potential's SCF expansion.
    particle_type : sequence of str, optional
        Species label(s) that were integrated (e.g. ``("STAR",)``).
    centring : str, optional
        The centring selector (see :meth:`ParticleSystem.recentre`) used to
        recentre the particle system before integration.
    binary_semimajor_axis : float, optional
        Semimajor axis of the central SMBH binary (physical length units), set
        by :func:`analyse_family` when the potential holds a bound two-BH
        binary; None otherwise. Drives :attr:`binary_interacting`.
    binary_interaction_factor : float, optional
        An orbit is binary-interacting if its pericentre is below this many
        binary semimajor axes (default 1). May be changed after integration;
        :attr:`binary_interacting` is recomputed on access.
    pattern_speed : numpy.ndarray, optional
        (3,) angular velocity of the figure the orbits were integrated in
        (physical units, velocity unit / length unit); all zero for a static
        potential (None also means static).
    orientation : str, optional
        How the axis signs were fixed before integration
        (:attr:`lanfear.ParticleSystem.orientation`): ``"pattern"`` or
        ``"spin"`` (+z along the pattern speed or the net spin, +x along the
        net ``L_x``), or None (arbitrary signs). It sets what a positive
        circulation sense means.
    pattern_speed_sigma : numpy.ndarray, optional
        (3,) 1-sigma uncertainty of each component of :attr:`pattern_speed`
        (same units), inherited from the potential
        (:attr:`lanfear.ParticleSystem.pattern_speed_sigma`). ``None`` when
        the pattern speed was not estimated (given explicitly, or static by
        choice).
    system_radius : float, optional
        Reference size of the integrated system (HO units): the 99.9th
        percentile of the orbits' initial radii, recorded by
        :func:`analyse_family`. Drives :attr:`runaway`; if None (results not
        built by :func:`analyse_family`) it is computed from
        :attr:`initial_radius` on access.
    runaway_factor : float, optional
        An orbit is a runaway if its maximum radius exceeds this many times
        the larger of its initial radius and :attr:`system_radius` (default
        10). May be changed after integration; :attr:`runaway` is recomputed
        on access.
    """

    ids: np.ndarray  # (N,) particle IDs
    summary: np.ndarray  # (N, len(SUMMARY_COLUMNS))
    columns: Sequence[str]  # column names for `summary`
    time_unit: float  # HO time -> physical time factor
    length_unit: float  # HO length -> physical length factor (scale radius)
    n_periods: int
    n_samples: int
    initial_radius: np.ndarray  # (N,) snapshot radius, HO units
    fundamentals: np.ndarray  # (N, 3) leading freq per axis
    lines: np.ndarray  # (N, 3, n_lines, 2) freq, amp
    diffusion: np.ndarray  # (N, 3) Laskar diffusion per axis
    planar_lines: np.ndarray  # (N, 3, n_lines, 2) signed planar lines
    n_periods_used: Optional[np.ndarray] = None  # (N,) periods integrated
    diffusion_previous: Optional[np.ndarray] = None  # (N,) rate before last extension
    source_file: Optional[str] = None  # particle-data file analysed
    n_max: Optional[int] = None  # SCF radial truncation order
    l_max: Optional[int] = None  # SCF angular truncation order
    particle_type: Optional[Sequence[str]] = None  # species integrated
    centring: Optional[str] = None  # how the system was recentred
    binary_semimajor_axis: Optional[float] = None  # physical length units
    binary_interaction_factor: float = 1.0  # r_peri threshold, in units of a
    pattern_speed: Optional[np.ndarray] = None  # (3,) figure angular velocity
    pattern_speed_sigma: Optional[np.ndarray] = None  # (3,) its 1-sigma error
    orientation: Optional[str] = None  # axis-sign reference ("pattern"/"spin")
    system_radius: Optional[float] = None  # HO units, 99.9th pct initial radius
    runaway_factor: float = 10.0  # r_max threshold, in units of the system size

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

        Raises
        ------
        KeyError
            If these results have no such column (e.g. they were written by a
            version of lanfear with another summary layout).
        """
        try:
            index = list(self.columns).index(name)
        except ValueError:
            raise KeyError(f"These results have no summary column {name!r}") from None
        return self.summary[:, index]

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
        """
        w = np.abs(self.fundamentals)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.stack([w[:, 0] / w[:, 2], w[:, 1] / w[:, 2]], axis=1)

    @property
    def binary_interacting(self) -> np.ndarray:
        """Boolean mask of orbits whose pericentre reaches the SMBH binary.

        An orbit is flagged when its pericentre (the ``r_peri`` summary column,
        measured from the origin -- the binary's centre of mass after
        ``prepare(centre="bh")``) is below ``binary_interaction_factor *
        binary_semimajor_axis``. Such an orbit would be scattered by the binary
        in reality, which no static potential represents, so its
        classification is not trustworthy; drop the flagged orbits from
        downstream analysis with :meth:`drop_binary_interacting` (or the
        ``drop_binary_interacting`` option of the classifiers).

        Returns
        -------
        mask : numpy.ndarray
            (N,) True where the orbit is binary-interacting. All False when no
            bound binary was recorded (:attr:`binary_semimajor_axis` is None
            or infinite).
        """
        a = self.binary_semimajor_axis
        if a is None or not np.isfinite(a):
            return np.zeros(len(self.ids), dtype=bool)
        r_peri = self.column("r_peri")
        return r_peri * self.length_unit < self.binary_interaction_factor * a

    @property
    def runaway(self) -> np.ndarray:
        """Boolean mask of orbits that run away from the system.

        An orbit is flagged when its maximum radius (``r_max``) exceeds
        ``runaway_factor`` times the larger of its initial radius and
        :attr:`system_radius`, i.e. it leaves the region the potential was
        fitted to and does not come back within the integration. Such orbits
        are unbound in the (possibly rotating) potential -- typically stars
        near or beyond corotation in a figure whose velocities were not drawn
        for that rotation -- so their family label is meaningless. The
        classifiers drop them by default (their ``drop_runaways`` option);
        :meth:`drop_runaways` removes them from the results themselves.

        Returns
        -------
        mask : numpy.ndarray
            (N,) True where the orbit runs away.
        """
        if len(self.ids) == 0:
            return np.zeros(0, dtype=bool)
        system_radius = self.system_radius
        if system_radius is None:
            system_radius = float(np.percentile(self.initial_radius, 99.9))
        reference = np.maximum(self.initial_radius, system_radius)
        return self.column("r_max") > self.runaway_factor * reference

    @property
    def body_periods_integrated(self) -> np.ndarray:
        """Number of body-frame periods each orbit was integrated for.

        ``t_total / body_period``. For a rotating figure, the window is
        lengthened (see :func:`analyse_family`'s ``max_body_period_factor``) so
        this is at least ``n_periods / sqrt(2)``, unless the orbit is so close
        to corotation that the lengthening hit its cap; such orbits are
        under-resolved in the co-rotating frame, and their frequencies (and so
        labels) are the least reliable. Equal to the inertial periods
        integrated for a static potential.

        Returns
        -------
        cycles : numpy.ndarray
            (N,) body-frame periods covered; 0 at exact corotation (infinite
            body-frame period), NaN for failed orbits.
        """
        body_period = np.asarray(self.column("body_period"), dtype=np.float64)
        t_total = np.asarray(self.column("t_total"), dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            cycles = t_total / body_period
        cycles[~self.ok] = np.nan
        return cycles

    # ------------------------------------------- frame-aware diagnostics
    @property
    def pattern_speed_ho(self) -> np.ndarray:
        """The pattern speed in HO units (rad / HO time).

        Returns
        -------
        omega : numpy.ndarray
            (3,) angular velocity in the units of :attr:`fundamentals`; all
            zero for a static potential (or when :attr:`pattern_speed` is None).
        """
        if self.pattern_speed is None:
            return np.zeros(3)
        return np.asarray(self.pattern_speed, dtype=np.float64) * self.time_unit

    def pattern_axis(self, max_tilt_deg: float = 5.0) -> Optional[Tuple[int, float]]:
        """The principal axis the figure rotates about, and its signed rate.

        The exact frame conversions (:meth:`azimuthal_frequency`,
        :meth:`azimuth_net`) hold for rotation about one body axis. The pattern
        speed of an aligned system lies along a principal axis up to the
        alignment noise (a fraction of a degree), which is ignored.

        Parameters
        ----------
        max_tilt_deg : float, optional
            Largest accepted angle between the pattern speed and its nearest
            principal axis (default 5 degrees).

        Returns
        -------
        axis : tuple of (int, float) or None
            ``(a, omega_a)``: the axis index (0=x, 1=y, 2=z) and the signed
            pattern speed about it (HO units); None for a static potential.

        Raises
        ------
        ValueError
            If the pattern speed is tilted from every principal axis by more
            than ``max_tilt_deg``.
        """
        omega = self.pattern_speed_ho
        speed = float(np.linalg.norm(omega))
        if speed == 0.0:
            return None
        axis = int(np.argmax(np.abs(omega)))
        tilt = np.degrees(np.arccos(min(abs(omega[axis]) / speed, 1.0)))
        if tilt > max_tilt_deg:
            raise ValueError(
                f"The pattern speed {np.round(omega, 4)} is {tilt:.1f} deg from the "
                f"nearest principal axis (more than max_tilt_deg={max_tilt_deg:g}); "
                f"the exact frame conversions need rotation about a principal axis"
            )
        return axis, float(omega[axis])

    @property
    def frequency_resolution(self) -> np.ndarray:
        """Frequency resolution ``2 pi / t_total`` of each orbit's window (HO).

        A frequency below about this cannot be told from zero: it completes
        less than one cycle in the window. Windows lengthened for a rotating
        figure (or extended for diffusion) have a finer resolution.

        Returns
        -------
        resolution : numpy.ndarray
            (N,) angular frequency, HO units; NaN for failed orbits.
        """
        t_total = np.asarray(self.column("t_total"), dtype=np.float64)
        with np.errstate(divide="ignore"):
            resolution = 2.0 * np.pi / t_total
        resolution[~self.ok] = np.nan
        return resolution

    def resolved(self, frequency, k: float = 1.0) -> np.ndarray:
        """Whether frequencies complete at least ``k`` cycles in each window.

        Parameters
        ----------
        frequency : array-like of float
            (N,) or (N, m) angular frequencies (HO units, any sign), one row
            per orbit; e.g. :meth:`azimuthal_frequency` or the
            ``radial_frequency`` column.
        k : float, optional
            Minimum number of cycles in the window (default 1; e.g. 3 for a
            frequency used to identify resonances).

        Returns
        -------
        mask : numpy.ndarray
            Same shape as ``frequency``; True where
            ``|frequency| >= k * frequency_resolution``. False for NaN
            frequencies and failed orbits.
        """
        frequency = np.asarray(frequency, dtype=np.float64)
        resolution = self.frequency_resolution
        if frequency.ndim == 2:
            resolution = resolution[:, None]
        with np.errstate(invalid="ignore"):
            return np.abs(frequency) >= k * resolution

    def azimuth_net(self, frame: str = "body") -> np.ndarray:
        """Signed angle swept about each principal axis over the window.

        The ``phi_x_net``/``phi_y_net``/``phi_z_net`` summary columns (body
        frame). In the inertial frame the angle about the pattern-speed axis
        grows by exactly ``Omega_p t_total``; about the other two axes it is not
        defined (the axes themselves turn), so those are NaN.

        Parameters
        ----------
        frame : {"body", "inertial"}, optional
            Frame of the angle (identical for a static potential).

        Returns
        -------
        phi : numpy.ndarray
            (N, 3) swept angle (rad); NaN for failed orbits.
        """
        names = [f"phi_{a}_net" for a in "xyz"]
        phi = np.stack(
            [np.asarray(self.column(n), dtype=np.float64) for n in names], axis=1
        )
        phi[~self.ok] = np.nan
        return self._to_frame(
            phi, frame, lambda omega_a: omega_a * self.column("t_total")
        )

    def winding_circulation(self) -> np.ndarray:
        """How steadily each orbit winds about each principal axis.

        ``|phi_net| / phi_range`` per axis (body frame): the swept angle over
        the spread of the unwrapped angle. An orbit that circulates about an
        axis winds monotonically, so the two are equal and the ratio is 1; a box
        or a librating orbit swings back and forth, so its net angle is a small
        fraction of the spread. Unlike the angular-momentum circulation it does
        not depend on how fast the orbit circulates, so a tube that the frame
        slows to a crawl near corotation still reads ~1 as long as it keeps
        going round. Close passages of the axis (boxes) can alias the
        unwrapping, which only adds noise to an already small ratio.

        In a tumbling figure a box also *precesses* steadily (Coriolis), and
        over many periods that precession is a monotonic winding too, so a
        precessing box can read close to 1 while passing the centre on every
        radial oscillation. Its winding is slow compared with its radial motion
        (``|azimuthal_frequency| / radial_frequency`` ~0.1, against ~0.5--1 for
        a tube) and it has no hole (:meth:`hole_fraction`); use those to tell
        the two apart.

        Returns
        -------
        circulation : numpy.ndarray
            (N, 3) ratio in [0, 1]; NaN for failed orbits or no angular motion.
        """
        net = np.stack([self.column(f"phi_{a}_net") for a in "xyz"], axis=1)
        spread = np.stack([self.column(f"phi_{a}_range") for a in "xyz"], axis=1)
        net = np.asarray(net, dtype=np.float64)
        spread = np.asarray(spread, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            circulation = np.where(spread > 0.0, np.abs(net) / spread, np.nan)
        circulation[~self.ok] = np.nan
        return np.minimum(circulation, 1.0)

    def hole_fraction(self) -> np.ndarray:
        """Minimum distance from each principal axis, relative to the orbit size.

        ``rho_a_min / r_mean`` per axis (``rho_*_min`` columns, body frame). A
        tube keeps clear of the axis it circulates about (a large hole); a box
        passes close to the centre, so all three are small. Holes alone do not
        identify tubes (resonant boxlets such as bananas avoid the centre too),
        but together with :meth:`winding_circulation` they separate a tube from
        a box that precesses steadily in a tumbling figure.

        Returns
        -------
        fraction : numpy.ndarray
            (N, 3) hole size in units of the mean radius; NaN for failed
            orbits.
        """
        rho = np.stack(
            [np.asarray(self.column(f"rho_{a}_min"), dtype=np.float64) for a in "xyz"],
            axis=1,
        )
        r_mean = np.asarray(self.column("r_mean"), dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = rho / r_mean[:, None]
        fraction[~self.ok] = np.nan
        return fraction

    def spectral_asymmetry(self) -> np.ndarray:
        """Amplitude asymmetry of each planar spectrum (diagnostic only).

        ``(A+ - A-) / (A+ + A-)`` with ``A+``/``A-`` the strongest positive-
        and negative-frequency lines of :attr:`planar_lines`. It is ~0 for a box
        (mirror-symmetric lines), but for a loop it measures the axis ratio of
        the loop rather than circulation, and the sidebands of a long-axis tube
        can dominate its circulation line, so it is *not* a circulation
        criterion; see :meth:`winding_circulation`.

        Returns
        -------
        asymmetry : numpy.ndarray
            (N, 3) values in [-1, 1]; NaN without planar lines of either sign.
        """
        planar = np.asarray(self.planar_lines, dtype=np.float64)
        freq, amp = planar[..., 0], planar[..., 1]
        plus = np.where(freq > 0, amp, 0.0).max(axis=2)
        minus = np.where(freq < 0, amp, 0.0).max(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (plus - minus) / (plus + minus)

    def _to_frame(self, values, frame, shift):
        """Convert (N, 3) per-axis body-frame values to the requested frame.

        Parameters
        ----------
        values : numpy.ndarray
            (N, 3) body-frame values, one column per axis.
        frame : {"body", "inertial"}
            Requested frame.
        shift : callable
            ``shift(omega_a)`` -> (N,) amount added to the pattern-axis column
            in the inertial frame, given the signed pattern speed ``omega_a``.

        Returns
        -------
        values : numpy.ndarray
            (N, 3) values; in the inertial frame of a rotating figure the
            pattern-axis column is shifted and the other two are NaN.
        """
        if frame not in ("body", "inertial"):
            raise ValueError(f"frame must be 'body' or 'inertial', got {frame!r}")
        axis = self.pattern_axis()
        if frame == "body" or axis is None:
            return values
        a, omega_a = axis
        out = np.full_like(values, np.nan)
        out[:, a] = values[:, a] + shift(omega_a)
        return out

    def azimuthal_frequency(self, frame: str = "body", k: float = 1.0) -> np.ndarray:
        """Signed circulation (azimuthal) frequency about each principal axis.

        The planar line (:attr:`planar_lines`) closest to the winding rate
        ``phi_net / t_total`` (:meth:`azimuth_net`). The winding rate locates
        the circulation line to within about half the frequency resolution;
        the line then gives its NAFF-refined value. Looking near the winding
        rate, rather than taking the strongest line, matters for long-axis
        tubes, whose planar spectra have sidebands stronger than the
        circulation line. Where no line lies within ``k`` frequency
        resolutions of the winding rate (boxes, librating orbits, slowly
        circulating orbits), the winding rate itself is returned.

        For a rotating figure the inertial-frame frequency about the
        pattern-speed axis is the body-frame one plus the signed pattern
        speed, exactly (the planar signal only gains the phase
        ``exp(i Omega_p t)``); about the other two axes it is not defined.

        Parameters
        ----------
        frame : {"body", "inertial"}, optional
            Frame of the frequency (identical for a static potential).
        k : float, optional
            Search radius around the winding rate, in frequency resolutions.

        Returns
        -------
        frequency : numpy.ndarray
            (N, 3) signed angular frequency (HO units; positive for winding in
            the sense of positive ``L`` about the axis); NaN for failed orbits
            and, in the inertial frame of a rotating figure, about the two axes
            other than the pattern-speed axis.
        """
        t_total = np.asarray(self.column("t_total"), dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = self.azimuth_net("body") / t_total[:, None]
        planar = np.asarray(self.planar_lines, dtype=np.float64)
        freq, amp = planar[..., 0], planar[..., 1]
        # Ignore empty (zero-amplitude) line slots.
        distance = np.where(amp > 0, np.abs(freq - rate[:, :, None]), np.inf)
        nearest = np.argmin(distance, axis=2)
        line = np.take_along_axis(freq, nearest[..., None], axis=2)[..., 0]
        gap = np.take_along_axis(distance, nearest[..., None], axis=2)[..., 0]
        with np.errstate(invalid="ignore"):
            close = gap <= k * self.frequency_resolution[:, None]
        frequency = np.where(close, line, rate)
        return self._to_frame(frequency, frame, lambda omega_a: omega_a)

    def pattern_speed_ratio(self, min_circulation: float = 0.5) -> np.ndarray:
        """Per-orbit ratio of the pattern speed to the orbit's own rotation.

        ``epsilon = |Omega_p| / |Omega_phi|`` with ``Omega_phi`` the
        inertial-frame azimuthal frequency about the pattern-speed axis
        (:meth:`azimuthal_frequency`). It measures how strongly the figure's
        rotation perturbs the orbit: ~0 for an orbit circulating much faster
        than the pattern, ~1 at corotation, large for an orbit the pattern
        outruns, and infinite for one locked to the inertial frame
        (:meth:`pattern_locked`). It is undefined for orbits that do not
        circulate about that axis in the body frame (boxes, and loops about
        another axis).

        Parameters
        ----------
        min_circulation : float, optional
            Orbits whose :meth:`winding_circulation` about the pattern-speed
            axis is below this get NaN (default 0.5).

        Returns
        -------
        ratio : numpy.ndarray
            (N,) ``epsilon``; all zero for a static potential, NaN for failed
            and non-circulating orbits.
        """
        n = len(self.ids)
        axis = self.pattern_axis()
        if axis is None:
            return np.zeros(n)
        a, omega_a = axis
        omega_phi = self.azimuthal_frequency("inertial")[:, a]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = abs(omega_a) / np.abs(omega_phi)
            circulating = self.winding_circulation()[:, a] >= min_circulation
        ratio[~circulating] = np.nan
        return ratio

    def pattern_locked(self, tol: float = 1e-3) -> np.ndarray:
        """Boolean mask of orbits that barely move in the inertial frame.

        In a rotating figure, an orbit much slower than the pattern (near or
        beyond corotation, or running away) hardly turns in the inertial
        frame, so in the co-rotating frame it just rotates backwards at
        ``-Omega_p``: its x and y share the frequency ``|Omega_p|``, which is
        the frame rotation, not a property of the orbit. It is flagged when its
        inertial azimuthal frequency about the pattern-speed axis
        (:meth:`azimuthal_frequency`) is below ``tol`` times ``|Omega_p|``.

        Parameters
        ----------
        tol : float, optional
            Locking tolerance, as a fraction of ``|Omega_p|`` (default 1e-3,
            in the gap between locked and unlocked orbits found on rotating
            test models).

        Returns
        -------
        mask : numpy.ndarray
            (N,) True where the orbit is pattern-locked; all False for a static
            potential.
        """
        n = len(self.ids)
        axis = self.pattern_axis()
        if axis is None:
            return np.zeros(n, dtype=bool)
        a, omega_a = axis
        omega_phi = self.azimuthal_frequency("inertial")[:, a]
        with np.errstate(invalid="ignore"):
            return np.abs(omega_phi) < tol * abs(omega_a)

    def select(self, mask) -> "OrbitResults":
        """Return a new OrbitResults holding only the selected orbits.

        Parameters
        ----------
        mask : numpy.ndarray
            (N,) boolean mask or integer index array selecting orbits.

        Returns
        -------
        results : OrbitResults
            The selected orbits, with all per-orbit arrays subset consistently
            and the metadata carried over.
        """
        mask = np.asarray(mask)
        return replace(
            self,
            ids=self.ids[mask],
            summary=self.summary[mask],
            initial_radius=self.initial_radius[mask],
            fundamentals=self.fundamentals[mask],
            lines=self.lines[mask],
            diffusion=self.diffusion[mask],
            planar_lines=self.planar_lines[mask],
            n_periods_used=(
                None if self.n_periods_used is None else self.n_periods_used[mask]
            ),
            diffusion_previous=(
                None
                if self.diffusion_previous is None
                else self.diffusion_previous[mask]
            ),
        )

    def drop_binary_interacting(self) -> "OrbitResults":
        """Return these results without the binary-interacting orbits.

        Returns
        -------
        results : OrbitResults
            A new OrbitResults excluding every orbit flagged by
            :attr:`binary_interacting` (all orbits if no binary was recorded).
        """
        flagged = self.binary_interacting
        logger.info(
            f"Dropping {int(flagged.sum())} of {len(flagged)} binary-interacting "
            f"orbits"
        )
        return self.select(~flagged)

    def drop_runaways(self) -> "OrbitResults":
        """Return these results without the runaway orbits.

        Returns
        -------
        results : OrbitResults
            A new OrbitResults excluding every orbit flagged by
            :attr:`runaway`. The system radius is fixed to the value used for
            the flag, so the remaining orbits stay unflagged.
        """
        flagged = self.runaway
        logger.info(f"Dropping {int(flagged.sum())} of {len(flagged)} runaway orbits")
        kept = self.select(~flagged)
        if kept.system_radius is None and len(self.ids):
            kept = replace(
                kept, system_radius=float(np.percentile(self.initial_radius, 99.9))
            )
        return kept

    def diffusion_rate(self, amp_frac: float = 0.05) -> np.ndarray:
        """Per-orbit Laskar frequency-diffusion rate (a chaos indicator).

        The rate of an axis is ``|w2 - w1| / |w1|``, comparing the leading
        frequency in the first and second halves of the integration. It is
        ~0 (numerical noise) for regular orbits and grows for chaotic ones. The
        per-orbit value is the maximum over the axes that actually oscillate:
        an axis whose leading-line amplitude is below ``amp_frac`` of the
        strongest axis is ignored, as its frequency is noise-dominated.

        Parameters
        ----------
        amp_frac : float, optional
            Relative amplitude below which an axis is treated as inactive.

        Returns
        -------
        rate : numpy.ndarray
            (N,) diffusion rate; NaN where no active axis has a valid value.
        """
        return _diffusion_rate(self.diffusion, self.lines, amp_frac)

    def to_dict(self) -> dict:
        """Flatten the results into a ``name -> array`` dictionary.

        Returns
        -------
        data : dict
            One entry per summary column plus ``"id"``,
            ``"freq_x"``/``"freq_y"``/``"freq_z"`` and
            ``"diffusion_x"``/``"diffusion_y"``/``"diffusion_z"``, and (when
            present) ``"n_periods_used"`` and ``"diffusion_previous"``, and
            ``"binary_interacting"`` (see :attr:`binary_interacting`),
            ``"runaway"`` (see :attr:`runaway`) and
            ``"body_periods_integrated"`` (see
            :attr:`body_periods_integrated`),
            ``"planar_freq_x"``/``"planar_freq_y"``/``"planar_freq_z"``
            (leading planar lines), ``"winding_circulation_*"`` and
            ``"azimuthal_frequency_*"`` (body frame; see
            :meth:`winding_circulation` and :meth:`azimuthal_frequency`), and
            ``"pattern_speed_ratio"`` and ``"pattern_locked"`` (default
            arguments).
        """
        d = {name: self.summary[:, i] for i, name in enumerate(self.columns)}
        d["id"] = self.ids
        d["freq_x"] = self.fundamentals[:, 0]
        d["freq_y"] = self.fundamentals[:, 1]
        d["freq_z"] = self.fundamentals[:, 2]
        if self.n_periods_used is not None:
            d["n_periods_used"] = self.n_periods_used
        if self.diffusion_previous is not None:
            d["diffusion_previous"] = self.diffusion_previous
        d["diffusion_x"] = self.diffusion[:, 0]
        d["diffusion_y"] = self.diffusion[:, 1]
        d["diffusion_z"] = self.diffusion[:, 2]
        winding = self.winding_circulation()
        azimuthal = self.azimuthal_frequency()
        for a, axis in enumerate("xyz"):
            d[f"planar_freq_{axis}"] = self.planar_lines[:, a, 0, 0]
            d[f"winding_circulation_{axis}"] = winding[:, a]
            d[f"azimuthal_frequency_{axis}"] = azimuthal[:, a]
        d["pattern_speed_ratio"] = self.pattern_speed_ratio()
        d["pattern_locked"] = self.pattern_locked()
        d["binary_interacting"] = self.binary_interacting
        d["runaway"] = self.runaway
        d["body_periods_integrated"] = self.body_periods_integrated
        return d

    def save(self, path: Union[str, os.PathLike]) -> str:
        """Write these results to a compact ``.npz`` file.

        Orbit integration is expensive, so this saves everything needed to
        rebuild the :class:`OrbitResults` (per-orbit summary, IDs, column names,
        the integration metadata, and the frequency data) into a
        single NumPy archive. Reload it with :meth:`load` to resume analysis
        (classification, plotting) without re-integrating.

        Provenance metadata is included when available: the source particle
        file (:attr:`source_file`), the SCF truncation orders
        (:attr:`n_max`/:attr:`l_max`), the integrated particle type
        (:attr:`particle_type`), the centring method (:attr:`centring`),
        the SMBH binary's semimajor axis and interaction factor
        (:attr:`binary_semimajor_axis`/:attr:`binary_interaction_factor`), and
        the figure's pattern speed and its uncertainty (:attr:`pattern_speed`/
        :attr:`pattern_speed_sigma`), and the runaway
        criterion (:attr:`system_radius`/:attr:`runaway_factor`).

        The archive is *uncompressed*, and the large per-orbit float arrays
        (``summary``, ``fundamentals``, ``lines``, ``diffusion``,
        ``planar_lines``, ``initial_radius``) are
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
            "fundamentals": np.asarray(self.fundamentals, dtype=np.float32),
            "lines": np.asarray(self.lines, dtype=np.float32),
            "diffusion": np.asarray(self.diffusion, dtype=np.float32),
            "planar_lines": np.asarray(self.planar_lines, dtype=np.float32),
            "binary_interaction_factor": np.asarray(
                self.binary_interaction_factor, dtype=np.float64
            ),
            "runaway_factor": np.asarray(self.runaway_factor, dtype=np.float64),
        }
        if self.n_periods_used is not None:
            arrays["n_periods_used"] = np.asarray(self.n_periods_used, dtype=np.int64)
        if self.diffusion_previous is not None:
            arrays["diffusion_previous"] = np.asarray(
                self.diffusion_previous, dtype=np.float32
            )
        if self.source_file is not None:
            arrays["source_file"] = np.asarray(self.source_file)
        if self.n_max is not None:
            arrays["n_max"] = np.asarray(self.n_max, dtype=np.int64)
        if self.l_max is not None:
            arrays["l_max"] = np.asarray(self.l_max, dtype=np.int64)
        if self.particle_type is not None:
            arrays["particle_type"] = np.asarray(list(self.particle_type))
        if self.centring is not None:
            arrays["centring"] = np.asarray(self.centring)
        if self.binary_semimajor_axis is not None:
            arrays["binary_semimajor_axis"] = np.asarray(
                self.binary_semimajor_axis, dtype=np.float64
            )
        if self.pattern_speed is not None:
            arrays["pattern_speed"] = np.asarray(self.pattern_speed, dtype=np.float64)
        if self.orientation is not None:
            arrays["orientation"] = np.asarray(self.orientation)
        if self.pattern_speed_sigma is not None:
            arrays["pattern_speed_sigma"] = np.asarray(
                self.pattern_speed_sigma, dtype=np.float64
            )
        if self.system_radius is not None:
            arrays["system_radius"] = np.asarray(self.system_radius, dtype=np.float64)
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
            The reconstructed results.

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
                fundamentals=npz["fundamentals"],
                lines=npz["lines"],
                diffusion=npz["diffusion"],
                planar_lines=npz["planar_lines"],
                n_periods_used=(
                    npz["n_periods_used"] if "n_periods_used" in npz else None
                ),
                diffusion_previous=(
                    npz["diffusion_previous"] if "diffusion_previous" in npz else None
                ),
                source_file=(str(npz["source_file"]) if "source_file" in npz else None),
                n_max=int(npz["n_max"]) if "n_max" in npz else None,
                l_max=int(npz["l_max"]) if "l_max" in npz else None,
                particle_type=(
                    tuple(str(s) for s in npz["particle_type"])
                    if "particle_type" in npz
                    else None
                ),
                centring=str(npz["centring"]) if "centring" in npz else None,
                binary_semimajor_axis=(
                    float(npz["binary_semimajor_axis"])
                    if "binary_semimajor_axis" in npz
                    else None
                ),
                binary_interaction_factor=float(npz["binary_interaction_factor"]),
                pattern_speed=(
                    np.asarray(npz["pattern_speed"], dtype=np.float64)
                    if "pattern_speed" in npz
                    else None
                ),
                orientation=(str(npz["orientation"]) if "orientation" in npz else None),
                pattern_speed_sigma=(
                    np.asarray(npz["pattern_speed_sigma"], dtype=np.float64)
                    if "pattern_speed_sigma" in npz
                    else None
                ),
                system_radius=(
                    float(npz["system_radius"]) if "system_radius" in npz else None
                ),
                runaway_factor=float(npz["runaway_factor"]),
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

    def classify_probabilistic(self, **kwargs):
        """Posterior-probability classification of these orbits.

        Parameters
        ----------
        **kwargs
            Passed through to
            :func:`lanfear.probabilistic_classify.classify_orbits_probabilistic`
            (e.g. ``model=`` to reuse a previously fitted/saved
            :class:`~lanfear.BayesianOrbitClassifier` instead of fitting one
            fresh).

        Returns
        -------
        classification : lanfear.ProbabilisticOrbitClassification
            Per-orbit MAP label, posterior probability and entropy.
        """
        from .probabilistic_classify import classify_orbits_probabilistic

        return classify_orbits_probabilistic(self, **kwargs)


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


def _unshuffle(rows, order):
    """Undo the scatter permutation of gathered rows.

    Parameters
    ----------
    rows : numpy.ndarray
        (N, ...) rows in permuted order: ``rows[k]`` belongs to orbit
        ``order[k]``.
    order : numpy.ndarray
        (N,) permutation applied before the scatter.

    Returns
    -------
    rows : numpy.ndarray
        The rows in the original orbit order.
    """
    out = np.empty_like(rows)
    out[order] = rows
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
    pattern_speed=None,
    max_body_period_factor: int = MAX_BODY_PERIOD_FACTOR,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """Integrate and frequency-analyse HO-unit states, MPI-distributed.

    Integrates each orbit and extracts the leading ``n_lines`` spectral lines
    per axis (the frequency data that drives classification), and as many
    signed lines of the planar signal about each axis. With a non-zero
    ``pattern_speed`` the potential rotates rigidly: orbits are integrated in
    the inertial frame and analysed in the co-rotating frame (see
    :mod:`lanfear.orbits`). Under MPI the orbits are shuffled (with a fixed
    permutation) before being split across ranks, so spatially ordered input
    does not leave one rank with all the expensive orbits; the outputs are
    returned in the input order.

    Parameters
    ----------
    scf : lanfear._core.SCFPotential or lanfear._core.DiscPotential
        The C++ potential; need only be valid on ``root`` (it is broadcast).
    states : numpy.ndarray or None
        (N, 6) inertial initial states in HO units; need only be valid on
        ``root``.
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
    pattern_speed : array-like of float, optional
        (3,) angular velocity of the figure in HO units (rad / HO time); need
        only be valid on ``root``. None (default) or zero for a static
        potential.
    max_body_period_factor : int, optional
        For a rotating figure, each orbit's window is lengthened by a power of
        two (at most this) so that it spans at least ``n_periods / sqrt(2)``
        body-frame periods as well as ``n_periods`` inertial ones, with
        ``n_samples`` raised alike (see
        :attr:`OrbitResults.body_periods_integrated`). 1 disables the
        lengthening.

    Returns
    -------
    summary : numpy.ndarray or None
        (N, len(SUMMARY_COLUMNS)) dynamics summary on ``root``.
    fundamentals : numpy.ndarray or None
        (N, 3) fundamental frequencies on ``root``.
    lines : numpy.ndarray or None
        (N, 3, n_lines, 2) leading spectral lines on ``root``.
    diffusion : numpy.ndarray or None
        (N, 3) Laskar frequency-diffusion rate per axis on ``root`` (see
        :attr:`OrbitResults.diffusion`).
    planar_lines : numpy.ndarray or None
        (N, 3, n_lines, 2) signed planar-signal lines on ``root`` (see
        :attr:`OrbitResults.planar_lines`). All five are None on non-root
        ranks.
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

    def _omega_tuple(value):
        """Pattern speed as the (3,) float tuple the C++ core takes.

        Parameters
        ----------
        value : array-like of float or None
            (3,) angular velocity in HO units, or None for no rotation.

        Returns
        -------
        omega : tuple of float
            The three components (all zero for None).
        """
        if value is None:
            return (0.0, 0.0, 0.0)
        return tuple(float(w) for w in np.asarray(value, dtype=np.float64).reshape(3))

    if comm is None:  # serial
        if states is None:
            raise ValueError("states must be provided for serial analysis")
        return scf.analyse_batch(
            np.ascontiguousarray(states, dtype=np.float64),
            n_periods,
            n_samples,
            abs_tol,
            rel_tol,
            n_lines,
            progress,
            _omega_tuple(pattern_speed),
            max_body_period_factor,
        )

    rank = comm.Get_rank()
    scf = comm.bcast(scf if rank == root else None, root=root)
    omega = comm.bcast(_omega_tuple(pattern_speed) if rank == root else None, root=root)
    if rank == root and states is None:
        raise ValueError("states must be provided on the root rank")

    # Shuffle the orbits before splitting them into per-rank blocks. Snapshots
    # are usually stored in spatial (e.g. Peano-Hilbert) order, so contiguous
    # blocks are patches of the galaxy, and the cost of an orbit depends on
    # where it is (most strongly near corotation, where rotating-figure orbits
    # are integrated for longer): unshuffled, one rank can get several times
    # the average work while the rest wait in the gather. The permutation is
    # fixed and inverted after the gather, so the results are unchanged.
    order = None
    if rank == root:
        order = np.random.default_rng(_SCATTER_SEED).permutation(len(states))
        states = np.asarray(states)[order]
    local_states, counts = _scatter_rows(
        comm, states if rank == root else np.empty((0, 6)), 6, root
    )
    l_summ, l_fund, l_lines, l_diff, l_planar = scf.analyse_batch(
        local_states,
        n_periods,
        n_samples,
        abs_tol,
        rel_tol,
        n_lines,
        progress and rank == root,
        omega,
        max_body_period_factor,
    )

    summary = _gather_rows(comm, l_summ, counts, ncol, root)
    fundamentals = _gather_rows(comm, l_fund, counts, 3, root)
    diffusion = _gather_rows(comm, l_diff, counts, 3, root)
    lines_flat = _gather_rows(
        comm, l_lines.reshape(len(l_lines), line_cols), counts, line_cols, root
    )
    planar_flat = _gather_rows(
        comm, l_planar.reshape(len(l_planar), line_cols), counts, line_cols, root
    )
    if rank == root:
        summary, fundamentals, diffusion, lines_flat, planar_flat = (
            _unshuffle(a, order)
            for a in (summary, fundamentals, diffusion, lines_flat, planar_flat)
        )
    return (
        summary,
        fundamentals,
        _reshape_lines(lines_flat),
        diffusion,
        _reshape_lines(planar_flat),
    )


def analyse_family(
    potential: AnyPotential,
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
    max_extensions: int = 0,
    extension_factor: float = 2.0,
    diffusion_threshold: float = 0.1,
    diffusion_drop: float = 0.5,
    binary_interaction_factor: float = 1.0,
    runaway_factor: float = 10.0,
    max_body_period_factor: int = MAX_BODY_PERIOD_FACTOR,
) -> Optional[OrbitResults]:
    """Integrate and frequency-analyse every particle of the given family.

    Selects the family from ``particles``, integrates each orbit in
    ``potential``, and returns an :class:`OrbitResults` carrying the per-orbit
    summary plus ``fundamentals`` (N, 3) and ``lines`` (N, 3, n_lines, 2) for
    resonance-based classification, and the Laskar frequency ``diffusion``
    (N, 3) (first- vs second-half frequency drift) as a chaos indicator.

    Optionally the integration is extended for orbits that drift in frequency, to
    tell chaos from a short window. Orbits whose diffusion rate exceeds
    ``diffusion_threshold`` are re-integrated ``extension_factor`` times longer.
    A quasi-periodic (regular) orbit's measured drift is a resolution effect and
    falls steeply (roughly as the square of the length) as the window grows,
    whereas a chaotic orbit's does not. An orbit whose rate has not fallen below
    ``diffusion_drop`` times its previous value is therefore chaotic and is not
    extended again; one that is still above the threshold but falling is
    extended further, up to ``max_extensions`` times. Orbits below the threshold
    are never re-integrated. The classification step
    (:func:`lanfear.classify.classify_orbits`) applies the same rule to label the
    chaotic orbits irregular.

    If ``potential`` rotates (a non-zero ``potential.pattern_speed``, inherited
    from the particle system, see :meth:`ParticleSystem.prepare`), each orbit is
    integrated in the inertial frame in the rigidly rotating potential and
    analysed in its co-rotating frame; the pattern speed is recorded on the
    results (see :mod:`lanfear.orbits`).

    If ``potential`` holds a bound SMBH binary (``potential.binary``, set by
    ``from_particles`` for a snapshot with two BHs), its semimajor axis is
    recorded on the results, and every orbit whose pericentre comes within
    ``binary_interaction_factor`` semimajor axes of the centre is flagged in
    :attr:`OrbitResults.binary_interacting`. The flagged orbits are kept; drop
    them from downstream analysis with
    :meth:`OrbitResults.drop_binary_interacting` or the classifiers'
    ``drop_binary_interacting`` option.

    Orbits that leave the system (maximum radius beyond ``runaway_factor``
    times the larger of their initial radius and the system radius, see
    :attr:`OrbitResults.runaway`) are likewise kept but flagged, with a
    warning. The classifiers leave them out by default (their
    ``drop_runaways`` option), and :meth:`OrbitResults.drop_runaways` removes
    them from the results. They are typically stars near or beyond corotation
    of a rotating figure.

    Parameters
    ----------
    potential : Potential, DiscPotential or MultiComponentPotential
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
    max_extensions : int, optional
        Maximum number of times to extend high-drift orbits. The default ``0``
        disables extension. Round ``k`` integrates the still-undecided orbits for
        ``n_periods * extension_factor**k`` periods.
    extension_factor : float, optional
        Factor (> 1) by which the integration length grows each round. The
        number of samples grows in proportion, keeping the sampling interval.
    diffusion_threshold : float, optional
        Orbits with a diffusion rate (:meth:`OrbitResults.diffusion_rate`) above
        this are extended. Use the value you will classify with; see
        ``scripts/choose_diffusion_threshold.py``.
    diffusion_drop : float, optional
        An extended orbit whose rate is at least this fraction of its previous
        rate is judged chaotic and not extended further. For a regular orbit the
        rate falls by roughly ``extension_factor**-2`` on extension, so this
        should sit well above that (0.5 for the default factor of 2).
    binary_interaction_factor : float, optional
        Orbits with pericentre below this many binary semimajor axes are
        flagged as binary-interacting (default 1). Ignored without a bound
        binary.
    runaway_factor : float, optional
        Orbits whose maximum radius exceeds this many times the larger of
        their initial radius and the system radius (99.9th percentile of the
        initial radii) are flagged as runaways (default 10).
    max_body_period_factor : int, optional
        For a rotating figure, an orbit's body-frame period
        ``2 pi / |Omega_c - |Omega_p||`` (summary column ``body_period``)
        exceeds its inertial period near corotation, where its co-rotating
        frequencies tend to zero and ``n_periods`` inertial periods would not
        resolve them. Each window is therefore lengthened by the power of
        two (at most this factor) nearest to ``body_period / period``, so that
        it spans at least ``n_periods / sqrt(2)`` body-frame periods, with
        ``n_samples`` raised by the same factor so the sampling interval is
        unchanged. Orbits with ``epsilon`` below ~0.3 are not lengthened.
        Orbits that hit the cap are reported (see
        :attr:`OrbitResults.body_periods_integrated`). 1 disables the
        lengthening; ignored for a static potential.

    Returns
    -------
    results : OrbitResults or None
        Per-orbit results (with frequency data) on ``root``; None on other ranks.
        When ``max_extensions > 0`` it also carries ``n_periods_used`` and
        ``diffusion_previous``.

    Raises
    ------
    ValueError
        On ``root`` if ``particles`` is None or no particles match ``family``, or
        on any rank if ``max_extensions`` > 0 and ``extension_factor`` <= 1 or
        ``diffusion_drop`` <= 0.
    """
    if max_extensions > 0:
        if not extension_factor > 1.0:
            raise ValueError(f"extension_factor must be > 1, got {extension_factor}")
        if not diffusion_drop > 0.0:
            raise ValueError(f"diffusion_drop must be > 0, got {diffusion_drop}")
    resolved = _resolve_comm(comm)
    rank = resolved.Get_rank() if resolved is not None else 0
    size = resolved.Get_size() if resolved is not None else 1

    states = ids = omega_ho = None
    if rank == root:
        if particles is None:
            raise ValueError("particles must be provided on the root rank")
        labels = (family,) if isinstance(family, str) else tuple(family)
        sub = particles.select(particles.species_mask(*labels))
        if sub.n_particles == 0:
            raise ValueError(f"no particles matched family {labels}")
        states = potential.to_ho_state(sub.pos, sub.vel)
        ids = sub.ids
        omega_ho = potential.pattern_speed_ho
        workers = f"{size} MPI ranks" if size > 1 else "1 process (serial)"
        frame = (
            f"figure rotating at {np.round(potential.pattern_speed, 4)}"
            if potential.rotating
            else "static potential"
        )
        logger.info(
            f"Integrating + frequency-analysing {sub.n_particles} orbits "
            f"(family={labels}) for {n_periods} periods on {workers} "
            f"({frame})"
        )

    t0 = time.perf_counter()
    analyse_kwargs = dict(
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        n_lines=n_lines,
        comm=resolved,
        root=root,
        progress=progress,
        pattern_speed=omega_ho,
        max_body_period_factor=max_body_period_factor,
    )
    summary, fundamentals, lines, diffusion, planar_lines = analyse_states(
        potential.core if rank == root else None,
        states,
        n_periods=n_periods,
        n_samples=n_samples,
        **analyse_kwargs,
    )

    n_used = previous = None
    if rank == root:
        n_used = np.full(len(states), n_periods, dtype=np.int64)
        previous = np.full(len(states), np.nan)  # rate before the last extension
    # Collective: every rank takes part in each round, root decides.
    for k in range(1, max_extensions + 1):
        flagged = rate = None
        if rank == root:
            rate = _diffusion_rate(diffusion, lines, 0.05)
            status_ok = summary[:, SUMMARY_COLUMNS.index("status")] == 0
            # Undecided orbits: above the threshold, and not already shown to be
            # failing to settle (rate not falling relative to the last length).
            # NaN rates compare False, so are never flagged.
            not_falling = rate >= diffusion_drop * previous
            flagged = status_ok & (rate > diffusion_threshold) & ~not_falling
        n_flagged = int(flagged.sum()) if rank == root else None
        if resolved is not None:
            n_flagged = resolved.bcast(n_flagged, root=root)
        if n_flagged == 0:
            break
        periods_k = int(np.ceil(n_periods * extension_factor**k))
        samples_k = int(round(n_samples * periods_k / n_periods))
        if rank == root:
            logger.info(
                f"Extension {k}/{max_extensions}: {n_flagged} of {len(flagged)} "
                f"orbits ({100 * n_flagged / len(flagged):.1f}%) have diffusion "
                f"rate > {diffusion_threshold:g}; re-integrating for {periods_k} "
                f"periods"
            )
        e_summ, e_fund, e_lines, e_diff, e_planar = analyse_states(
            potential.core if rank == root else None,
            states[flagged] if rank == root else None,
            n_periods=periods_k,
            n_samples=samples_k,
            **analyse_kwargs,
        )
        if rank == root:
            previous[flagged] = rate[flagged]
            summary[flagged] = e_summ
            fundamentals[flagged] = e_fund
            lines[flagged] = e_lines
            diffusion[flagged] = e_diff
            planar_lines[flagged] = e_planar
            n_used[flagged] = periods_k
    if rank == root and max_extensions > 0:
        rate = _diffusion_rate(diffusion, lines, 0.05)
        high = np.isfinite(previous) & (rate > diffusion_threshold)
        n_chaotic = int(np.sum(high & (rate >= diffusion_drop * previous)))
        n_open = int(np.sum(high)) - n_chaotic
        logger.info(
            f"Extension finished: {int(np.sum(np.isfinite(previous)))} orbits "
            f"extended, {n_chaotic} drifting with the window (chaotic)"
        )
        if n_open:
            logger.warning(
                f"{n_open} orbits are still above the diffusion threshold but "
                f"falling after {max_extensions} extension(s); raise "
                f"max_extensions to settle them"
            )

    if rank != root:
        return None
    _log_integration_result(summary, time.perf_counter() - t0)
    binary = getattr(potential, "binary", None)
    binary_a = binary.semimajor_axis if binary is not None and binary.bound else None
    initial_radius = np.linalg.norm(states[:, :3], axis=1)
    results = OrbitResults(
        ids=np.asarray(ids),
        summary=summary,
        columns=SUMMARY_COLUMNS,
        time_unit=potential.time_unit,
        n_periods=n_periods,
        n_samples=n_samples,
        fundamentals=fundamentals,
        lines=lines,
        diffusion=diffusion,
        planar_lines=planar_lines,
        n_periods_used=n_used if max_extensions > 0 else None,
        diffusion_previous=previous if max_extensions > 0 else None,
        length_unit=potential.scale_radius,
        initial_radius=initial_radius,
        source_file=particles.source_file,
        n_max=potential.n_max,
        l_max=potential.l_max,
        particle_type=labels,
        centring=particles.centring,
        binary_semimajor_axis=binary_a,
        binary_interaction_factor=binary_interaction_factor,
        pattern_speed=potential.pattern_speed,
        pattern_speed_sigma=potential.pattern_speed_sigma,
        orientation=particles.orientation,
        system_radius=(
            float(np.percentile(initial_radius, 99.9)) if len(initial_radius) else None
        ),
        runaway_factor=runaway_factor,
    )
    if potential.rotating:
        periods = n_used if max_extensions > 0 else n_periods
        ok = results.ok
        n_ok = max(int(ok.sum()), 1)
        lengthened = ok & (
            results.column("t_total") > 1.5 * periods * results.column("period")
        )
        capped = ok & (results.body_periods_integrated < periods / np.sqrt(2))
        logger.info(
            f"{int(lengthened.sum())} orbits ({100 * lengthened.sum() / n_ok:.1f}%) "
            f"were integrated for longer to span ~{n_periods} body-frame periods"
        )
        if capped.any():
            logger.warning(
                f"{int(capped.sum())} orbits ({100 * capped.sum() / n_ok:.1f}%) "
                f"are so close to corotation that even "
                f"{max_body_period_factor} x the window spans fewer than "
                f"{n_periods}/sqrt(2) body-frame periods; their frequencies are "
                f"under-resolved (raise max_body_period_factor, or see "
                f"OrbitResults.body_periods_integrated)"
            )
    n_runaway = int(results.runaway.sum())
    if n_runaway:
        logger.warning(
            f"{n_runaway} of {len(results.ids)} orbits "
            f"({100 * n_runaway / len(results.ids):.1f}%) run away beyond "
            f"{runaway_factor:g} x the system radius (flagged runaway; "
            f"classification leaves them out by default)"
        )
    if binary_a is not None:
        n_flagged = int(results.binary_interacting.sum())
        logger.info(
            f"{n_flagged} of {len(results.ids)} orbits "
            f"({100 * n_flagged / len(results.ids):.1f}%) reach within "
            f"{binary_interaction_factor:g} x a = "
            f"{binary_interaction_factor * binary_a:.4g} of the SMBH binary "
            f"(flagged binary_interacting)"
        )
    return results


class ParticleTrajectory:
    """The phase-space trajectory of a single particle.

    Keeping the full trajectory of every particle in an :class:`OrbitResults`
    batch "just in case" would be wasteful; in practice only a handful of
    orbits are ever inspected in this much detail. This class instead
    re-integrates one particle on demand (via
    ``potential.core.integrate_orbit``, the single-orbit counterpart of the
    batch integrator) and keeps only that trajectory, in physical units.

    For a rotating potential (non-zero ``potential.pattern_speed``) the orbit is
    integrated in the inertial frame but recorded in the co-rotating frame of
    the figure, where the potential is static; for a static potential the two
    frames coincide. As in :func:`analyse_family`, a rotating figure's window
    near corotation is lengthened (by up to 8x, with the samples scaled alike)
    to span the requested number of body-frame periods, so the arrays may hold
    a multiple of ``n_samples`` samples.

    Parameters
    ----------
    particle_id : int or None
        Identifier of the particle this trajectory belongs to.
    pos : numpy.ndarray
        (n_samples, 3) co-rotating-frame positions in physical length units,
        uniformly sampled in time.
    vel : numpy.ndarray
        (n_samples, 3) co-rotating-frame velocities ``d pos / dt`` in physical
        velocity units, sampled at the same times as ``pos``.
    energy : numpy.ndarray
        (n_samples,) Jacobi integral ``0.5 |vel|^2 + Phi - 0.5 |Omega x pos|^2``
        (the specific orbital energy ``0.5 |v|^2 + Phi`` for a static
        potential) in physical units (velocity unit squared, e.g. (km/s)^2),
        sampled at the same times as ``pos``. It is conserved along the orbit.
    time : numpy.ndarray
        (n_samples,) physical times at which ``pos`` was sampled.
    status : int
        Integrator status (0 ok, 1 period estimate failed, 2 NaN
        encountered); ``pos``/``vel``/``energy``/``time`` may be short or
        empty when non-zero.
    pattern_speed : numpy.ndarray, optional
        (3,) angular velocity of the figure (physical units); all zero
        (default) for a static potential.
    """

    def __init__(
        self,
        particle_id: Optional[int],
        pos: np.ndarray,
        vel: np.ndarray,
        energy: np.ndarray,
        time: np.ndarray,
        status: int,
        pattern_speed: Optional[np.ndarray] = None,
    ) -> None:
        self.particle_id = particle_id
        self.pos = pos
        self.vel = vel
        self.energy = energy
        self.time = time
        self.status = status
        self.pattern_speed = (
            np.zeros(3) if pattern_speed is None else np.asarray(pattern_speed)
        )

    @classmethod
    def integrate(
        cls,
        potential: AnyPotential,
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
        potential : Potential, DiscPotential or MultiComponentPotential
            The analytical potential to integrate in (rotating at its
            ``pattern_speed``).
        pos_phys : array-like of float
            (3,) particle position in physical length units.
        vel_phys : array-like of float
            (3,) particle (inertial) velocity in physical velocity units.
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
        omega_ho = potential.pattern_speed_ho
        summary, traj_ho = potential.core.integrate_orbit(
            state,
            n_periods=n_periods,
            n_samples=n_samples,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            return_trajectory=True,
            pattern_speed=tuple(float(w) for w in omega_ho),
        )
        status = int(summary[_COL_INDEX["status"]])
        if status != 0:
            logger.warning(
                f"Particle {particle_id} did not integrate cleanly "
                f"(status={status})"
            )

        # Samples are uniformly spaced at dt_out = t_total / (n_samples' - 1),
        # where n_samples' = n_samples * factor if a rotating figure's window
        # was lengthened by `factor` to span n_periods body-frame periods.
        # Reconstruct time from that spacing (rather than assuming n_samples'
        # rows) so a trajectory truncated by a mid-integration NaN still gets
        # correctly timed samples.
        t_total_ho = summary[_COL_INDEX["t_total"]]
        period_ho = summary[_COL_INDEX["period"]]
        factor = 1
        if period_ho > 0:
            factor = max(int(round(t_total_ho / (n_periods * period_ho))), 1)
        dt_out_ho = t_total_ho / (n_samples * factor - 1)
        time = dt_out_ho * np.arange(len(traj_ho)) * potential.time_unit
        pos = traj_ho[:, :3] * potential.scale_radius
        vel = traj_ho[:, 3:] * potential.velocity_unit

        # Jacobi integral (the specific energy when static) from the sampled
        # co-rotating states, evaluated in HO units (where the integrator
        # works) and scaled to physical units by V^2. The body-frame potential
        # is static, so it is evaluated at the co-rotating positions directly.
        if len(traj_ho):
            phi_ho = potential.core.potential_batch(
                np.ascontiguousarray(traj_ho[:, :3])
            )
            kinetic_ho = 0.5 * np.einsum("ij,ij->i", traj_ho[:, 3:], traj_ho[:, 3:])
            spin = np.cross(omega_ho, traj_ho[:, :3])  # Omega x x_b
            centrifugal_ho = 0.5 * np.einsum("ij,ij->i", spin, spin)
            energy = (kinetic_ho + phi_ho - centrifugal_ho) * potential.velocity_unit**2
        else:
            energy = np.empty(0)
        return cls(
            particle_id=particle_id,
            pos=pos,
            vel=vel,
            energy=energy,
            time=time,
            status=status,
            pattern_speed=potential.pattern_speed,
        )

    @classmethod
    def from_particles(
        cls,
        potential: AnyPotential,
        particles: ParticleSystem,
        particle_id: int,
        **kwargs,
    ) -> "ParticleTrajectory":
        """Integrate the trajectory of one particle picked out of a system.

        Parameters
        ----------
        potential : Potential, DiscPotential or MultiComponentPotential
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

    def plot_energy(self, ax=None, relative: bool = True, **plot_kwargs):
        """Plot the orbit's specific energy (Jacobi integral) against time.

        The plotted quantity is :attr:`energy`: the specific energy for a
        static potential, or the Jacobi integral for a rotating one. Either is
        conserved, so any trend in this plot is integration error; it is the
        per-sample view of the ``energy_drift`` summary column.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. A new figure is created if omitted.
        relative : bool, optional
            If True (default), plot the relative drift ``(E - E0) / |E0|``
            (the quantity whose maximum magnitude is ``energy_drift``);
            otherwise plot the energy ``E`` itself, in physical units.
        **plot_kwargs
            Passed through to :meth:`matplotlib.axes.Axes.plot`.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes drawn on.

        Raises
        ------
        ValueError
            If the trajectory has fewer than 2 samples.
        """
        if len(self.energy) < 2:
            raise ValueError("trajectory has fewer than 2 samples to plot")

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 3.3), layout="constrained")

        if relative:
            energy_0 = self.energy[0]
            values = (self.energy - energy_0) / abs(energy_0)
            ylabel = r"$(E - E_0)\,/\,|E_0|$"
        else:
            values = self.energy
            ylabel = r"$E$"
        ax.plot(self.time, values, **plot_kwargs)
        ax.set_xlabel("time")
        ax.set_ylabel(ylabel)
        ax.set_title(f"ID: {self.particle_id}")
        return ax
