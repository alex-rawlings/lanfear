"""Rotation sense of orbits relative to a chosen reference spin.

A *reference* is the net angular momentum of a set of particles: the whole
field, one family (e.g. ``"STAR"``), or a family within some radius. Its
direction ``n`` is measured from the snapshot (:func:`reference_spin`, also
:meth:`lanfear.ParticleSystem.reference_spin`) and stored as a
:class:`SpinReference`.

Each orbit is compared with it through the inertial angular-momentum moments
``<L>`` and ``<L L^T>`` (the ``Lin_*`` summary columns), so a reference can be
added after integration:

* **coherence** ``kappa = |<L>| / sqrt(tr <L L^T>)`` in [0, 1]: whether the
  orbit has a net circulation at all (about 0 for boxes);
* **cosine** ``<L> . n / |<L>|``: which way it circulates relative to ``n``;
* **sign consistency** ``<L> . n / sqrt(n^T <L L^T> n)``: whether ``L . n``
  keeps its sign. A diagnostic only, since it does not measure alignment.

A *local* reference (:func:`reference_spin_profile`, a
:class:`ProfileReference`) measures the spin in radial shells instead, and
compares each orbit with the direction at its own orbit-averaged radius.
:class:`ReferenceSpec` holds either kind for later, and parses the compact
form used on command lines (``STAR:5``, ``DM``, ``STAR:local``).

:meth:`lanfear.OrbitResults.rotation_sense` turns these into a
:class:`RotationSense` label. This is independent of the loop ``sense`` of
:func:`lanfear.classify.classify_orbits`, which stays the default and is about
each loop's own circulation axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ._logging import get_logger
from .particle_system import Frame, _gaussian_window, _windowed_principal_axes

logger = get_logger(__name__)

# Calibrated on the static perfect ellipsoid (docs/tuning.rst).
DEFAULT_COHERENCE_MIN = 0.2
DEFAULT_PERPENDICULAR = 0.5  # |cos theta| below this: 60 to 120 degrees
DEFAULT_Z_MAX_FACTOR = 3.0  # default z_max, in units of the median |z|
DEFAULT_PROFILE_WIDTH = 0.35  # shell width of local references, in ln r

_GEOMETRIES = ("auto", "spherical", "cylindrical")
_WINDOWS = ("hard", "gaussian")


class RotationSense(IntEnum):
    """Sense of an orbit relative to a :class:`SpinReference`.

    ``NONE``: no net circulation (coherence below the threshold) or a failed
    orbit. ``PERPENDICULAR``: circulates about an axis roughly perpendicular to
    the reference. ``UNDEFINED``: the reference's own spin is not significant.
    """

    NONE = 0
    CO = 1
    COUNTER = 2
    PERPENDICULAR = 3
    UNDEFINED = 4


ROTATION_SENSE_NAMES = {s.value: s.name.lower() for s in RotationSense}


@dataclass(frozen=True, eq=False)
class SpinReference:
    """The net spin of a set of particles, used as a rotation-sense reference.

    Build with :func:`reference_spin`. Vectors are in the analysis frame
    :attr:`frame`, in the physical units of the particle system.

    Parameters
    ----------
    name : str
        Label, e.g. ``"field"``, ``"DM"``, ``"STAR_r<5"``.
    axis : numpy.ndarray
        (3,) unit vector along the net angular momentum (NaN if it is zero).
    net_angular_momentum : numpy.ndarray
        (3,) ``sum w m x x v`` over the selection (``w`` the window weight).
    sigma : numpy.ndarray
        (3,) Poisson uncertainty of each component.
    angle_uncertainty : float
        Uncertainty of the direction of :attr:`axis` (radians).
    significant : bool
        Whether the net spin exceeds ``significance`` times its uncertainty
        along :attr:`axis`.
    n_particles : int
        Number of particles selected.
    frame : Frame
        The particles' frame (:attr:`lanfear.ParticleSystem.frame`).
    family : tuple of str, optional
        Species selected; ``None`` for the whole field.
    radius_inner, radius_outer : float, optional
        Aperture radii (``radius_outer`` ``None``: no aperture).
    geometry : str, optional
        ``"spherical"`` or ``"cylindrical"``; ``None`` without an aperture.
    z_max : float, optional
        Half-height of a cylindrical aperture.
    cylinder_axis : numpy.ndarray, optional
        (3,) axis of a cylindrical aperture: the selection's own short axis,
        signed along :attr:`net_angular_momentum`.
    window : str
        ``"hard"`` or ``"gaussian"``.
    significance : float
        Threshold used for :attr:`significant`.
    """

    name: str
    axis: np.ndarray
    net_angular_momentum: np.ndarray
    sigma: np.ndarray
    angle_uncertainty: float
    significant: bool
    n_particles: int
    frame: Frame
    family: Optional[Tuple[str, ...]] = None
    radius_inner: float = 0.0
    radius_outer: Optional[float] = None
    geometry: Optional[str] = None
    z_max: Optional[float] = None
    cylinder_axis: Optional[np.ndarray] = None
    window: str = "hard"
    significance: float = 3.0

    @classmethod
    def from_axis(cls, axis, frame: Frame, name: str) -> "SpinReference":
        """A reference along a given direction rather than a measured spin.

        For example the pattern-speed axis, or +z of the analysis frame. It is
        always significant and has no uncertainty.

        Parameters
        ----------
        axis : array-like of float
            (3,) direction in the analysis frame (need not be normalised).
        frame : Frame
            The frame ``axis`` is given in (e.g. ``ps.frame``).
        name : str
            Label.

        Returns
        -------
        reference : SpinReference
            The reference, with ``n_particles = 0``.

        Raises
        ------
        ValueError
            If ``axis`` is zero or not finite.
        """
        axis = np.asarray(axis, dtype=np.float64)
        size = float(np.linalg.norm(axis))
        if not (np.isfinite(size) and size > 0.0):
            raise ValueError(f"axis must be a non-zero finite vector, got {axis!r}")
        return cls(
            name=name,
            axis=axis / size,
            net_angular_momentum=axis / size,
            sigma=np.zeros(3),
            angle_uncertainty=0.0,
            significant=True,
            n_particles=0,
            frame=frame,
            significance=0.0,
        )

    def axes_for(self, results) -> Tuple[np.ndarray, np.ndarray]:
        """The reference direction for every orbit of ``results``.

        Parameters
        ----------
        results : lanfear.OrbitResults
            The orbits.

        Returns
        -------
        axes : numpy.ndarray
            (N, 3) :attr:`axis` for every orbit.
        significant : numpy.ndarray
            (N,) :attr:`significant` for every orbit.
        """
        n = len(results.ids)
        return np.broadcast_to(self.axis, (n, 3)), np.full(n, self.significant)

    def _to_arrays(self, prefix: str) -> dict:
        """Flatten into ``npz`` arrays whose keys start with ``prefix``."""
        out = {
            "kind": np.asarray("spin"),
            "name": np.asarray(self.name),
            "axis": np.asarray(self.axis, dtype=np.float64),
            "net_angular_momentum": np.asarray(
                self.net_angular_momentum, dtype=np.float64
            ),
            "sigma": np.asarray(self.sigma, dtype=np.float64),
            "angle_uncertainty": np.asarray(self.angle_uncertainty, dtype=np.float64),
            "significant": np.asarray(self.significant),
            "n_particles": np.asarray(self.n_particles, dtype=np.int64),
            "frame_centre": self.frame.centre,
            "frame_velocity": self.frame.velocity,
            "frame_rotation": self.frame.rotation,
            "radius_inner": np.asarray(self.radius_inner, dtype=np.float64),
            "window": np.asarray(self.window),
            "significance": np.asarray(self.significance, dtype=np.float64),
        }
        if self.family is not None:
            out["family"] = np.asarray(list(self.family))
        for key in ("radius_outer", "z_max"):
            if getattr(self, key) is not None:
                out[key] = np.asarray(getattr(self, key), dtype=np.float64)
        if self.geometry is not None:
            out["geometry"] = np.asarray(self.geometry)
        if self.cylinder_axis is not None:
            out["cylinder_axis"] = np.asarray(self.cylinder_axis, dtype=np.float64)
        return {f"{prefix}{key}": value for key, value in out.items()}

    @classmethod
    def _from_arrays(cls, npz, prefix: str) -> "SpinReference":
        """Rebuild from the arrays written by :meth:`_to_arrays`."""

        def get(key, convert=lambda v: v):
            full = f"{prefix}{key}"
            return convert(npz[full]) if full in npz else None

        return cls(
            name=str(npz[f"{prefix}name"]),
            axis=np.asarray(npz[f"{prefix}axis"]),
            net_angular_momentum=np.asarray(npz[f"{prefix}net_angular_momentum"]),
            sigma=np.asarray(npz[f"{prefix}sigma"]),
            angle_uncertainty=float(npz[f"{prefix}angle_uncertainty"]),
            significant=bool(npz[f"{prefix}significant"]),
            n_particles=int(npz[f"{prefix}n_particles"]),
            frame=Frame(
                npz[f"{prefix}frame_centre"],
                npz[f"{prefix}frame_velocity"],
                npz[f"{prefix}frame_rotation"],
            ),
            family=get("family", lambda v: tuple(str(s) for s in v)),
            radius_inner=float(npz[f"{prefix}radius_inner"]),
            radius_outer=get("radius_outer", float),
            geometry=get("geometry", str),
            z_max=get("z_max", float),
            cylinder_axis=get("cylinder_axis", np.asarray),
            window=str(npz[f"{prefix}window"]),
            significance=float(npz[f"{prefix}significance"]),
        )


@dataclass(frozen=True)
class ReferenceSpec:
    """A reference to build later, against particles (and a potential).

    Holds the arguments of :func:`reference_spin` (or, with ``local=True``, of
    :func:`reference_spin_profile`); :meth:`resolve` builds the reference.
    :meth:`parse` reads the compact form used on command lines.

    ``radius`` and ``window`` apply to a single reference, ``radii`` and
    ``width`` to a local one; giving one to the other kind raises
    ``ValueError``.

    Parameters
    ----------
    family : str or sequence of str, optional
        Species to select; ``None`` for the whole field.
    radius : float or (float, float), optional
        Aperture radius, or ``(inner, outer)``.
    geometry : {"auto", "spherical", "cylindrical"}, optional
        Aperture shape (see :func:`reference_spin`).
    z_max : float, optional
        Half-height of a cylindrical aperture.
    window : {"hard", "gaussian"}, optional
        Aperture weighting.
    name : str, optional
        Label; generated if omitted.
    local : bool, optional
        Build a local reference (:func:`reference_spin_profile`) instead.
    radii : sequence of float, optional
        Shell radii of a local reference.
    width : float, optional
        Shell width of a local reference (in ``ln r``).
    """

    family: Optional[Union[str, Sequence[str]]] = None
    radius: Optional[Union[float, Tuple[float, float]]] = None
    geometry: str = "auto"
    z_max: Optional[float] = None
    window: str = "hard"
    name: Optional[str] = None
    local: bool = False
    radii: Optional[Tuple[float, ...]] = None
    width: float = DEFAULT_PROFILE_WIDTH

    def __post_init__(self) -> None:
        """Reject options the chosen kind of reference does not use."""
        if self.local and (self.radius is not None or self.window != "hard"):
            raise ValueError("a local reference takes no radius or window")
        if not self.local and (
            self.radii is not None or self.width != DEFAULT_PROFILE_WIDTH
        ):
            raise ValueError("radii and width apply to a local reference only")

    @classmethod
    def parse(cls, text: str) -> "ReferenceSpec":
        """Read a reference from its compact text form.

        ``FAMILY[:OPTION]...``, where ``FAMILY`` is a species label, several
        joined by ``+``, or ``field``, and each option is one of:

        * a radius ``5`` or a range ``2-8`` (aperture);
        * ``spherical`` or ``cylindrical`` (geometry), ``gaussian`` (window);
        * ``z=1.0`` (``z_max``), ``name=...``;
        * ``local`` (a local reference), ``width=0.35`` (its shell width).

        Examples: ``field``, ``DM``, ``STAR:5``, ``STAR+DM:2-8:spherical``,
        ``STAR:5:cylindrical:z=1.0``, ``STAR:local``.

        Parameters
        ----------
        text : str
            The compact form.

        Returns
        -------
        spec : ReferenceSpec
            The parsed spec.

        Raises
        ------
        ValueError
            If an option is not recognised or repeated, or does not apply to
            the kind of reference (see the class).
        """
        head, *options = text.split(":")
        if not head:
            raise ValueError(f"reference {text!r} names no family")
        family = None if head == "field" else tuple(head.split("+"))
        kwargs = {}

        def put(key, value):
            if key in kwargs:
                raise ValueError(f"reference {text!r} sets {key} twice")
            kwargs[key] = value

        for option in options:
            key, _, value = option.partition("=")
            if option in ("spherical", "cylindrical"):
                put("geometry", option)
            elif option == "gaussian":
                put("window", option)
            elif option == "local":
                put("local", True)
            elif key == "z" and value:
                put("z_max", float(value))
            elif key == "width" and value:
                put("width", float(value))
            elif key == "name" and value:
                put("name", value)
            else:
                try:
                    bounds = [float(v) for v in option.split("-")]
                except ValueError:
                    raise ValueError(
                        f"unrecognised option {option!r} in reference {text!r}"
                    ) from None
                if len(bounds) not in (1, 2):
                    raise ValueError(f"bad radius {option!r} in reference {text!r}")
                put("radius", bounds[0] if len(bounds) == 1 else tuple(bounds))
        try:
            return cls(family=family, **kwargs)
        except ValueError as exc:
            raise ValueError(f"{exc}: {text!r}") from None

    def resolve(self, particles, potential=None):
        """Build the reference from ``particles``.

        Parameters
        ----------
        particles : ParticleSystem
            The prepared particles, in the frame of the orbits.
        potential : Potential, DiscPotential or MultiComponentPotential, optional
            Resolves ``geometry="auto"``.

        Returns
        -------
        reference : SpinReference or ProfileReference
            The measured reference (a :class:`ProfileReference` if
            :attr:`local`).
        """
        if self.local:
            return reference_spin_profile(
                particles,
                family=self.family,
                radii=self.radii,
                width=self.width,
                geometry=self.geometry,
                z_max=self.z_max,
                name=self.name,
                potential=potential,
            )
        return reference_spin(
            particles,
            family=self.family,
            radius=self.radius,
            geometry=self.geometry,
            z_max=self.z_max,
            window=self.window,
            name=self.name,
            potential=potential,
        )


def _as_families(family) -> Optional[Tuple[str, ...]]:
    """Normalise ``family`` to a tuple of labels (or None)."""
    if family is None:
        return None
    if isinstance(family, str):
        return (family,)
    families = tuple(str(f) for f in family)
    if not families:
        raise ValueError("family must name at least one species")
    return families


def _selection(particles, families, mask, name) -> np.ndarray:
    """Boolean mask of the particles a reference is measured from."""
    if families is None:
        selected = np.asarray(particles.species) != "BH"
    else:
        selected = particles.species_mask(*families)
    if mask is not None:
        if name is None:
            raise ValueError("a reference with a custom mask needs a name")
        selected = selected & np.asarray(mask, dtype=bool)
    if not np.any(selected):
        raise ValueError(f"no particles match family={families!r}")
    return selected


def _spin_sigma(contribution: np.ndarray) -> np.ndarray:
    """Poisson uncertainty of ``contribution.sum(axis=0)``."""
    n = len(contribution)
    if n < 2:
        return np.full(3, np.inf)
    return np.sqrt(n * np.var(contribution, axis=0, ddof=1))


def _as_radii(radius) -> Tuple[float, Optional[float]]:
    """Normalise ``radius`` to ``(inner, outer)``."""
    if radius is None:
        return 0.0, None
    if np.ndim(radius) == 0:
        inner, outer = 0.0, float(radius)
    else:
        inner, outer = (float(r) for r in radius)
    if not (0.0 <= inner < outer):
        raise ValueError(f"radius must satisfy 0 <= inner < outer, got {radius!r}")
    return inner, outer


def _auto_geometry(potential, families: Optional[Tuple[str, ...]]) -> str:
    """Aperture shape from how each selected family was fitted.

    A family fitted with a disc basis is cylindrical, any other spherical.
    """
    if potential is None:
        raise ValueError(
            "geometry='auto' needs the potential to tell disc families from "
            "spheroidal ones; pass potential=..., or geometry='spherical' or "
            "'cylindrical'"
        )
    from .disc_potential import DiscPotential
    from .multi_component_potential import MultiComponentPotential

    labels = families
    if isinstance(potential, MultiComponentPotential):
        labels = families if families is not None else tuple(potential.components)
        missing = [label for label in labels if label not in potential.components]
        if missing:
            raise ValueError(
                f"geometry='auto': no fitted component for {missing} "
                f"(have {sorted(potential.components)})"
            )
        discs = {
            potential.components[label].kind == "DiscPotential" for label in labels
        }
    else:
        discs = {isinstance(potential, DiscPotential)}
    if len(discs) > 1:
        raise ValueError(
            f"geometry='auto': the selected families {labels} mix disc and "
            f"spheroidal fits; pass geometry='spherical' or 'cylindrical'"
        )
    return "cylindrical" if discs.pop() else "spherical"


def _cylindrical_selection(pos, mass, inner, outer, z_max, max_iterations):
    """Cylinder about the selection's own short axis.

    Starts from the spherical aperture, then alternates between taking the
    short axis of ``sum m x x^T`` and re-cutting the cylinder about it, until
    the axis and the half-height stop changing (a few particles at the edge
    may still swap in and out).

    Returns
    -------
    keep : numpy.ndarray
        (N,) boolean selection.
    axis : numpy.ndarray
        (3,) cylinder axis (sign arbitrary).
    z_cut : float
        Half-height used.
    """
    r = np.linalg.norm(pos, axis=1)
    keep = (r >= inner) & (r < outer)
    axis = None
    previous_cut = np.nan
    for _ in range(max_iterations):
        if not np.any(keep):
            raise ValueError("the cylindrical aperture selects no particles")
        _, axes = _windowed_principal_axes(pos[keep], mass[keep])
        new_axis = axes[:, 2]
        height = pos @ new_axis
        cyl_r = np.linalg.norm(pos - height[:, None] * new_axis, axis=1)
        in_ring = (cyl_r >= inner) & (cyl_r < outer)
        if z_max is None:
            if not np.any(in_ring):
                raise ValueError("the cylindrical aperture selects no particles")
            z_cut = DEFAULT_Z_MAX_FACTOR * float(np.median(np.abs(height[in_ring])))
        else:
            z_cut = float(z_max)
        new_keep = in_ring & (np.abs(height) < z_cut)
        converged = (
            axis is not None
            and abs(float(new_axis @ axis)) > 1.0 - 1e-12
            and abs(z_cut - previous_cut) <= 1e-6 * z_cut
        )
        axis, keep, previous_cut = new_axis, new_keep, z_cut
        if converged:
            return keep, axis, z_cut
    logger.warning(
        f"The cylindrical aperture did not converge in {max_iterations} iterations"
    )
    return keep, axis, z_cut


def _aperture_label(geometry, window, inner, outer, z_cut) -> str:
    """Name suffix describing the aperture."""
    if outer is None:
        return ""
    if window == "gaussian":
        return f"_gauss{outer:g}"
    coordinate = "R" if geometry == "cylindrical" else "r"
    label = (
        f"_{coordinate}<{outer:g}"
        if inner == 0.0
        else f"_{inner:g}<={coordinate}<{outer:g}"
    )
    if geometry == "cylindrical":
        label += f"_|z|<{z_cut:.3g}"
    return label


def reference_spin(
    particles,
    family: Optional[Union[str, Sequence[str]]] = None,
    radius: Optional[Union[float, Tuple[float, float]]] = None,
    geometry: str = "auto",
    z_max: Optional[float] = None,
    window: str = "hard",
    mask: Optional[np.ndarray] = None,
    name: Optional[str] = None,
    potential=None,
    significance: float = 3.0,
    max_iterations: int = 10,
) -> SpinReference:
    """Measure the net spin of a set of particles as a reference.

    Selects the whole field (``family=None``) or the given families, optionally
    within an aperture, and returns the direction of their net angular
    momentum about the origin.

    Parameters
    ----------
    particles : ParticleSystem
        The prepared particles, in the frame the orbits are analysed in.
    family : str or sequence of str, optional
        Species to select; ``None`` (default) for every field (non-BH)
        particle.
    radius : float or (float, float), optional
        Aperture radius, or ``(inner, outer)`` for a shell or annulus, in the
        particles' physical length units. ``None`` (default): no aperture.
    geometry : {"auto", "spherical", "cylindrical"}, optional
        Aperture shape. ``"spherical"`` selects ``inner <= r < outer``.
        ``"cylindrical"`` selects ``inner <= R < outer`` and ``|z| < z_max``
        about the selection's own short axis (found iteratively, so a tilted
        disc is cut correctly). ``"auto"`` (default) is cylindrical for
        families fitted with a disc basis in ``potential`` and spherical
        otherwise; it needs ``potential``. Only ``"auto"`` is allowed
        without an aperture.
    z_max : float, optional
        Half-height of a cylindrical aperture. Defaults to 3 times the median
        ``|z|`` of the selection with ``inner <= R < outer``.
    window : {"hard", "gaussian"}, optional
        ``"hard"`` (default) for the aperture above; ``"gaussian"`` weights
        every particle by ``exp(-r^2 / (2 radius^2))`` instead (spherical, no
        inner radius).
    mask : numpy.ndarray, optional
        (N,) extra boolean selection, combined with the others. Needs ``name``.
    name : str, optional
        Label; generated from the selection if omitted.
    potential : Potential, DiscPotential or MultiComponentPotential, optional
        Resolves ``geometry="auto"``.
    significance : float, optional
        A net spin below this many times its uncertainty along the axis is
        flagged as not significant, with a warning (default 3).
    max_iterations : int, optional
        Iterations of the cylindrical aperture.

    Returns
    -------
    reference : SpinReference
        The measured reference.

    Raises
    ------
    ValueError
        If an argument is not recognised, the selection is empty, ``mask`` is
        given without ``name``, ``geometry``, ``z_max`` or ``window`` is set
        without an aperture, or ``geometry="auto"`` cannot be resolved.
    """
    if geometry not in _GEOMETRIES:
        raise ValueError(f"geometry must be one of {_GEOMETRIES}, got {geometry!r}")
    if window not in _WINDOWS:
        raise ValueError(f"window must be one of {_WINDOWS}, got {window!r}")
    families = _as_families(family)
    inner, outer = _as_radii(radius)
    if outer is None and (geometry != "auto" or z_max is not None or window != "hard"):
        raise ValueError("geometry, z_max and window need an aperture radius")
    selected = _selection(particles, families, mask, name)
    pos = particles.pos[selected]
    vel = particles.vel[selected]
    mass = particles.mass[selected]

    shape = None
    z_cut = None
    cylinder_axis = None
    weight = np.ones(len(mass))
    if outer is not None:
        if window == "gaussian":
            if inner > 0.0 or geometry == "cylindrical":
                raise ValueError("window='gaussian' is spherical, with no inner radius")
            shape = "spherical"
            weight, _ = _gaussian_window(np.linalg.norm(pos, axis=1), outer)
        else:
            shape = (
                _auto_geometry(potential, families) if geometry == "auto" else geometry
            )
            if shape == "spherical":
                if z_max is not None:
                    raise ValueError(
                        "z_max applies to a cylindrical aperture only; pass "
                        "geometry='cylindrical'"
                    )
                r = np.linalg.norm(pos, axis=1)
                keep = (r >= inner) & (r < outer)
            else:
                keep, cylinder_axis, z_cut = _cylindrical_selection(
                    pos, mass, inner, outer, z_max, max_iterations
                )
            if not np.any(keep):
                raise ValueError(
                    f"no particles of family={family!r} in the {shape} aperture"
                )
            pos, vel, mass, weight = pos[keep], vel[keep], mass[keep], weight[keep]

    contribution = (mass * weight)[:, None] * np.cross(pos, vel)
    total = contribution.sum(axis=0)
    n = len(contribution)
    sigma = _spin_sigma(contribution)
    size = float(np.linalg.norm(total))
    if size > 0.0:
        axis = total / size
        sigma_along = float(np.sqrt(np.sum(sigma**2 * axis**2)))
        sigma_across = float(np.sqrt(np.sum(sigma**2 * (1.0 - axis**2))))
        angle = float(np.arctan2(sigma_across, size))
        significant = bool(size > significance * sigma_along)
    else:
        axis = np.full(3, np.nan)
        angle = float(np.pi)
        significant = False
    if cylinder_axis is not None and size > 0.0 and cylinder_axis @ total < 0:
        cylinder_axis = -cylinder_axis

    if name is None:
        base = "field" if families is None else "+".join(families)
        name = base + _aperture_label(shape, window, inner, outer, z_cut)
    if not significant:
        logger.warning(
            f"The net spin of reference '{name}' is consistent with zero "
            f"(< {significance:g} sigma); rotation senses relative to it are "
            f"undefined"
        )
    logger.debug(
        f"Reference '{name}': {n} particles, axis {np.round(axis, 4)} "
        f"+/- {np.degrees(angle):.2g} deg"
    )
    return SpinReference(
        name=name,
        axis=axis,
        net_angular_momentum=total,
        sigma=sigma,
        angle_uncertainty=angle,
        significant=significant,
        n_particles=n,
        frame=particles.frame,
        family=families,
        radius_inner=inner,
        radius_outer=outer,
        geometry=shape,
        z_max=z_cut,
        cylinder_axis=cylinder_axis,
        window=window,
        significance=float(significance),
    )


@dataclass(frozen=True, eq=False)
class ProfileReference:
    """A local reference: the net spin of a set of particles in radial shells.

    Build with :func:`reference_spin_profile`. Each orbit is compared with the
    reference direction at its own orbit-averaged radius (``r_mean``),
    interpolated between the shells, so an orbit is "co-rotating" if it turns
    with the particles around it. Vectors are in the analysis frame
    :attr:`frame`, in the physical units of the particle system.

    Parameters
    ----------
    name : str
        Label, e.g. ``"STAR_local"``.
    radii : numpy.ndarray
        (K,) shell radii, increasing.
    net_angular_momentum : numpy.ndarray
        (K, 3) window-weighted ``sum g m x x v`` of each shell.
    sigma : numpy.ndarray
        (K, 3) Poisson uncertainty of each component.
    significant : numpy.ndarray
        (K,) whether each shell's spin exceeds ``significance`` times its
        uncertainty along its axis.
    n_particles : int
        Number of particles selected.
    frame : Frame
        The particles' frame (:attr:`lanfear.ParticleSystem.frame`).
    family : tuple of str, optional
        Species selected; ``None`` for the whole field.
    geometry : str
        ``"spherical"`` (shells in ``r``) or ``"cylindrical"`` (in ``R``, with
        ``|z| < z_max`` about :attr:`cylinder_axis`).
    z_max : float, optional
        Half-height of the cylindrical selection.
    cylinder_axis : numpy.ndarray, optional
        (3,) axis of the cylindrical selection.
    width : float
        Shell width in ``ln r``.
    significance : float
        Threshold used for :attr:`significant`.
    """

    name: str
    radii: np.ndarray
    net_angular_momentum: np.ndarray
    sigma: np.ndarray
    significant: np.ndarray
    n_particles: int
    frame: Frame
    family: Optional[Tuple[str, ...]] = None
    geometry: str = "spherical"
    z_max: Optional[float] = None
    cylinder_axis: Optional[np.ndarray] = None
    width: float = DEFAULT_PROFILE_WIDTH
    significance: float = 3.0

    @property
    def axes(self) -> np.ndarray:
        """(K, 3) unit spin direction of each shell (NaN where it is zero)."""
        size = np.linalg.norm(self.net_angular_momentum, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(size > 0, self.net_angular_momentum / size, np.nan)

    @property
    def angle_uncertainty(self) -> np.ndarray:
        """(K,) uncertainty of each shell's direction (radians)."""
        axes = np.nan_to_num(self.axes)
        across = np.sqrt(np.sum(self.sigma**2 * (1.0 - axes**2), axis=1))
        return np.arctan2(across, np.linalg.norm(self.net_angular_momentum, axis=1))

    def axes_at(self, radius) -> Tuple[np.ndarray, np.ndarray]:
        """Reference direction and its significance at the given radii.

        The net angular momentum and its variance are interpolated linearly
        in ``ln r`` between the shells (constant beyond the first and last),
        so the direction turns smoothly through a radius where the spin
        flips, and is not significant there.

        Parameters
        ----------
        radius : array-like of float
            (N,) radii (physical length units).

        Returns
        -------
        axes : numpy.ndarray
            (N, 3) unit directions (NaN where the interpolated spin is zero).
        significant : numpy.ndarray
            (N,) whether the interpolated spin exceeds ``significance`` times
            its uncertainty along it.
        """
        log_r = np.log(self.radii)
        x = np.log(
            np.clip(np.asarray(radius, dtype=np.float64), self.radii[0], self.radii[-1])
        )
        spin = np.stack(
            [np.interp(x, log_r, self.net_angular_momentum[:, a]) for a in range(3)],
            axis=1,
        )
        variance = np.stack(
            [np.interp(x, log_r, self.sigma[:, a] ** 2) for a in range(3)], axis=1
        )
        size = np.linalg.norm(spin, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            axes = np.where(size[:, None] > 0, spin / size[:, None], np.nan)
        along = np.sqrt(np.sum(variance * np.nan_to_num(axes) ** 2, axis=1))
        significant = (size > 0) & (size > self.significance * along)
        return axes, significant

    def axes_for(self, results) -> Tuple[np.ndarray, np.ndarray]:
        """The reference direction at every orbit's orbit-averaged radius.

        Parameters
        ----------
        results : lanfear.OrbitResults
            The orbits.

        Returns
        -------
        axes, significant : numpy.ndarray
            As for :meth:`axes_at`, at ``r_mean`` of each orbit.
        """
        radius = results.column("r_mean").astype(np.float64) * results.length_unit
        return self.axes_at(radius)

    def _to_arrays(self, prefix: str) -> dict:
        """Flatten into ``npz`` arrays whose keys start with ``prefix``."""
        out = {
            "kind": np.asarray("profile"),
            "name": np.asarray(self.name),
            "radii": np.asarray(self.radii, dtype=np.float64),
            "net_angular_momentum": np.asarray(
                self.net_angular_momentum, dtype=np.float64
            ),
            "sigma": np.asarray(self.sigma, dtype=np.float64),
            "significant": np.asarray(self.significant, dtype=bool),
            "n_particles": np.asarray(self.n_particles, dtype=np.int64),
            "frame_centre": self.frame.centre,
            "frame_velocity": self.frame.velocity,
            "frame_rotation": self.frame.rotation,
            "geometry": np.asarray(self.geometry),
            "width": np.asarray(self.width, dtype=np.float64),
            "significance": np.asarray(self.significance, dtype=np.float64),
        }
        if self.family is not None:
            out["family"] = np.asarray(list(self.family))
        if self.z_max is not None:
            out["z_max"] = np.asarray(self.z_max, dtype=np.float64)
        if self.cylinder_axis is not None:
            out["cylinder_axis"] = np.asarray(self.cylinder_axis, dtype=np.float64)
        return {f"{prefix}{key}": value for key, value in out.items()}

    @classmethod
    def _from_arrays(cls, npz, prefix: str) -> "ProfileReference":
        """Rebuild from the arrays written by :meth:`_to_arrays`."""

        def get(key, convert=lambda v: v):
            full = f"{prefix}{key}"
            return convert(npz[full]) if full in npz else None

        return cls(
            name=str(npz[f"{prefix}name"]),
            radii=np.asarray(npz[f"{prefix}radii"]),
            net_angular_momentum=np.asarray(npz[f"{prefix}net_angular_momentum"]),
            sigma=np.asarray(npz[f"{prefix}sigma"]),
            significant=np.asarray(npz[f"{prefix}significant"], dtype=bool),
            n_particles=int(npz[f"{prefix}n_particles"]),
            frame=Frame(
                npz[f"{prefix}frame_centre"],
                npz[f"{prefix}frame_velocity"],
                npz[f"{prefix}frame_rotation"],
            ),
            family=get("family", lambda v: tuple(str(s) for s in v)),
            geometry=str(npz[f"{prefix}geometry"]),
            z_max=get("z_max", float),
            cylinder_axis=get("cylinder_axis", np.asarray),
            width=float(npz[f"{prefix}width"]),
            significance=float(npz[f"{prefix}significance"]),
        )


AnyReference = Union[SpinReference, ProfileReference]


def _reference_from_arrays(npz, prefix: str) -> AnyReference:
    """Rebuild a reference of either kind saved with ``_to_arrays``."""
    kind = str(npz[f"{prefix}kind"])
    return (ProfileReference if kind == "profile" else SpinReference)._from_arrays(
        npz, prefix
    )


def reference_spin_profile(
    particles,
    family: Optional[Union[str, Sequence[str]]] = None,
    radii=None,
    width: float = DEFAULT_PROFILE_WIDTH,
    geometry: str = "auto",
    z_max: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
    name: Optional[str] = None,
    potential=None,
    significance: float = 3.0,
    max_iterations: int = 10,
) -> ProfileReference:
    """Measure the net spin of a set of particles in radial shells.

    A local reference (:class:`ProfileReference`): each shell is a log-normal
    window ``g = exp(-ln(r / r_s)^2 / (2 width^2))`` about its radius ``r_s``,
    as in :meth:`lanfear.ParticleSystem.pattern_speed_profile`, so the profile
    follows a spin axis that turns or flips with radius (a warp, a
    counter-rotating core).

    Parameters
    ----------
    particles : ParticleSystem
        The prepared particles, in the frame the orbits are analysed in.
    family : str or sequence of str, optional
        Species to select; ``None`` (default) for every field particle.
    radii : array-like of float, optional
        Shell radii (physical length units). Defaults to 12 radii spaced
        evenly in ``ln r`` from the 10th to the 90th percentile of the
        selection's radii.
    width : float, optional
        Shell width in ``ln r`` (default 0.35).
    geometry : {"auto", "spherical", "cylindrical"}, optional
        ``"spherical"`` shells in ``r``; ``"cylindrical"`` shells in ``R``
        about the selection's own short axis, keeping ``|z| < z_max`` (as for
        :func:`reference_spin`). ``"auto"`` (default) is cylindrical for
        families fitted with a disc basis in ``potential``, spherical
        otherwise, and needs ``potential``.
    z_max : float, optional
        Half-height of the cylindrical selection. Defaults to 3 times the
        median ``|z|`` of the selection.
    mask : numpy.ndarray, optional
        (N,) extra boolean selection. Needs ``name``.
    name : str, optional
        Label; ``"<family>_local"`` if omitted.
    potential : Potential, DiscPotential or MultiComponentPotential, optional
        Resolves ``geometry="auto"``.
    significance : float, optional
        Threshold for each shell's (and each orbit's interpolated)
        significance (default 3).
    max_iterations : int, optional
        Iterations of the cylindrical selection.

    Returns
    -------
    reference : ProfileReference
        The measured profile.

    Raises
    ------
    ValueError
        If an argument is not recognised, the selection is empty, ``radii``
        is not positive and increasing, or ``geometry="auto"`` cannot be
        resolved.
    """
    if geometry not in _GEOMETRIES:
        raise ValueError(f"geometry must be one of {_GEOMETRIES}, got {geometry!r}")
    if not width > 0:
        raise ValueError(f"width must be positive, got {width}")
    families = _as_families(family)
    selected = _selection(particles, families, mask, name)
    pos = particles.pos[selected]
    vel = particles.vel[selected]
    mass = particles.mass[selected]
    shape = _auto_geometry(potential, families) if geometry == "auto" else geometry

    cylinder_axis = z_cut = None
    if shape == "cylindrical":
        keep, cylinder_axis, z_cut = _cylindrical_selection(
            pos, mass, 0.0, np.inf, z_max, max_iterations
        )
        pos, vel, mass = pos[keep], vel[keep], mass[keep]
        height = pos @ cylinder_axis
        coordinate = np.linalg.norm(pos - height[:, None] * cylinder_axis, axis=1)
    else:
        if z_max is not None:
            raise ValueError(
                "z_max applies to a cylindrical profile only; pass geometry='cylindrical'"
            )
        coordinate = np.linalg.norm(pos, axis=1)

    if radii is None:
        positive = coordinate[coordinate > 0]
        radii = np.geomspace(*np.percentile(positive, [10, 90]), 12)
    radii = np.atleast_1d(np.asarray(radii, dtype=np.float64))
    if np.any(radii <= 0) or np.any(np.diff(radii) <= 0):
        raise ValueError("radii must be positive and increasing")

    spin_per_particle = mass[:, None] * np.cross(pos, vel)
    with np.errstate(divide="ignore"):
        log_r = np.log(coordinate)
    totals = np.empty((len(radii), 3))
    sigma = np.empty((len(radii), 3))
    for k, radius in enumerate(radii):
        g = np.nan_to_num(np.exp(-0.5 * ((log_r - np.log(radius)) / width) ** 2))
        contribution = g[:, None] * spin_per_particle
        totals[k] = contribution.sum(axis=0)
        sigma[k] = _spin_sigma(contribution)
    size = np.linalg.norm(totals, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        axes = np.where(size[:, None] > 0, totals / size[:, None], 0.0)
    along = np.sqrt(np.sum(sigma**2 * axes**2, axis=1))
    significant = (size > 0) & (size > significance * along)
    if cylinder_axis is not None and cylinder_axis @ totals.sum(axis=0) < 0:
        cylinder_axis = -cylinder_axis

    if name is None:
        name = ("field" if families is None else "+".join(families)) + "_local"
    if not significant.any():
        logger.warning(
            f"No shell of local reference '{name}' has a significant net spin; "
            f"rotation senses relative to it are undefined"
        )
    elif not significant.all():
        logger.info(
            f"Local reference '{name}': the spin is not significant at radii "
            f"{np.round(radii[~significant], 3)}"
        )
    return ProfileReference(
        name=name,
        radii=radii,
        net_angular_momentum=totals,
        sigma=sigma,
        significant=significant,
        n_particles=len(mass),
        frame=particles.frame,
        family=families,
        geometry=shape,
        z_max=z_cut,
        cylinder_axis=cylinder_axis,
        width=float(width),
        significance=float(significance),
    )
