"""Multi-species potential: an independently-fit basis per particle species,
linearly superposed -- e.g. a stellar disc embedded in a dark-matter halo.

A single basis (the spheroidal SCF of :class:`~lanfear.Potential`, or the disc
basis of :class:`~lanfear.DiscPotential`) fits one shape well. A galaxy built
from physically distinct components -- a flattened stellar disc inside a
round-ish dark-matter halo, say -- is not well represented by either alone.
:class:`MultiComponentPotential` instead fits each species its own basis and
adds the results together: Poisson's equation is linear, so the sum of the
components' physical potentials/accelerations is the exact potential/
acceleration of the combined system.

Unit bookkeeping
-----------------
:class:`~lanfear.Potential` and :class:`~lanfear.DiscPotential` each work
internally in their own Hernquist-Ostriker-style units: ``G = (that
component's field mass) = (that component's scale radius) = 1``. Two
components built independently are therefore expressed on different physical
scales and cannot simply be added. This class picks one shared composite unit
system -- length unit ``L0`` (default: the half-mass radius of every species
combined) and mass unit ``M0`` (the combined field mass) -- and rescales each
component into it. For component ``i`` with its own scale radius ``a_i`` and
field mass ``M_i``:

    coord_scale_i = L0 / a_i
    phi_weight_i  = (M_i / M0) * coord_scale_i
    acc_weight_i  = (M_i / M0) * coord_scale_i ** 2

so that, writing ``x_h`` for a composite-HO coordinate,

    phi_h(x_h)   = sum_i phi_weight_i * component_i.potential(x_h * coord_scale_i)
    accel_h(x_h) = sum_i acc_weight_i * component_i.acceleration(x_h * coord_scale_i)

reproduce the composite-HO-normalised sum of the components' physical fields
(see ``include/lanfear/composite_potential.hpp`` for the same derivation on
the C++ side, where the weighted sum is actually evaluated).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from . import _core
from ._logging import get_logger
from ._potential_base import _PotentialBase
from .disc_potential import DiscPotential
from .particle_system import ParticleSystem
from .potential import Potential, ValidationResult

logger = get_logger(__name__)


@dataclass(frozen=True)
class ComponentSpec:
    """Which potential type (and build kwargs) to fit to one species.

    Not constructed directly; use :func:`scf_component` or
    :func:`disc_component`.

    Parameters
    ----------
    builder : type
        :class:`~lanfear.Potential` or :class:`~lanfear.DiscPotential` -- the
        class whose ``from_particles`` builds this component.
    kwargs : dict
        Keyword arguments forwarded to ``builder.from_particles`` (excluding
        ``particles`` and ``G``, which :class:`MultiComponentPotential`
        supplies itself).
    """

    builder: Callable
    kwargs: dict

    def build(self, particles: ParticleSystem, G: float):
        """Fit this component to one species' particles.

        Parameters
        ----------
        particles : ParticleSystem
            The (single-species) particles to fit.
        G : float
            Gravitational constant in the physical unit system.

        Returns
        -------
        component : Potential or DiscPotential
            The fitted single-species potential.
        """
        return self.builder.from_particles(particles, G=G, **self.kwargs)


def scf_component(**kwargs) -> ComponentSpec:
    """A Hernquist-Ostriker SCF component spec, for a spherical-ish species.

    Parameters
    ----------
    **kwargs
        Forwarded to :meth:`Potential.from_particles`; ``n_max`` and ``l_max``
        are required -- there is no default truncation order.

    Returns
    -------
    spec : ComponentSpec
        Declarative spec for :meth:`MultiComponentPotential.from_particles`.

    Raises
    ------
    TypeError
        If ``n_max`` or ``l_max`` is not given.
    """
    if "n_max" not in kwargs or "l_max" not in kwargs:
        raise TypeError("scf_component requires explicit n_max and l_max")
    return ComponentSpec(builder=Potential, kwargs=kwargs)


def disc_component(**kwargs) -> ComponentSpec:
    """A Miyamoto-Nagai disc-basis component spec, for a flattened species.

    Parameters
    ----------
    **kwargs
        Forwarded to :meth:`DiscPotential.from_particles` (e.g. ``n_radial``,
        ``n_vert``, or explicit ``a_scales``/``b_scales``).

    Returns
    -------
    spec : ComponentSpec
        Declarative spec for :meth:`MultiComponentPotential.from_particles`.
    """
    return ComponentSpec(builder=DiscPotential, kwargs=kwargs)


@dataclass(frozen=True)
class ComponentInfo:
    """Provenance of one fitted species component.

    Parameters
    ----------
    kind : str
        The component class name (``"Potential"`` or ``"DiscPotential"``).
    scale_radius : float
        The component's own (physical) scale radius.
    field_mass : float
        The component's own (physical) field mass.
    n_particles : int
        Number of particles of this species that were fit.
    """

    kind: str
    scale_radius: float
    field_mass: float
    n_particles: int


class MultiComponentPotential(_PotentialBase):
    """Linear superposition of an independently-fit potential per species.

    Build with :meth:`from_particles`, giving one :class:`ComponentSpec` (see
    :func:`scf_component`/:func:`disc_component`) per particle species present
    in the snapshot -- there is no default potential type, since silently
    guessing wrong (e.g. treating a disc as spherical) would produce a bad fit
    without warning. The result interfaces with the rest of lanfear exactly
    like a single-component potential (:attr:`core`, :meth:`to_ho_state`,
    :meth:`potential`/:meth:`acceleration`, orbit integration/analysis,
    :meth:`validate`, :meth:`plot_potential_plane`), since Poisson's equation
    is linear and the composite is just the (unit-reconciled) sum of the
    components -- see the module docstring for the unit bookkeeping.

    Example
    -------
    ::

        pot = lf.MultiComponentPotential.from_particles(
            ps,
            components={
                "STAR": lf.disc_component(n_radial=8, n_vert=3),
                "DM": lf.scf_component(n_max=18, l_max=7),
            },
        )
    """

    def __init__(
        self,
        core: "_core.CompositePotential",
        length_unit: float,
        field_mass: float,
        field_pos_ho: np.ndarray,
        field_mass_ho: np.ndarray,
        G: float,
        components: Dict[str, ComponentInfo],
        sub_potentials: Dict[str, _PotentialBase],
    ) -> None:
        """Wrap a built C++ composite potential and record its unit system.

        Most callers should use :meth:`from_particles` rather than
        constructing this directly.

        Parameters
        ----------
        core : lanfear._core.CompositePotential
            The built C++ composite (each component already weighted).
        length_unit : float
            Physical length used as the shared (composite) HO length unit.
        field_mass : float
            Total physical mass of every species' field particles combined;
            the shared HO mass unit.
        field_pos_ho : numpy.ndarray
            (N, 3) every species' field-particle positions, in the composite's
            HO units (kept for validation).
        field_mass_ho : numpy.ndarray
            (N,) every species' field-particle masses, in the composite's HO
            units (kept for validation).
        G : float, optional
            Gravitational constant in the physical unit system.
        components : dict of str to ComponentInfo
            Per-species build provenance, for introspection/logging.
        sub_potentials : dict of str to (Potential or DiscPotential)
            The fitted per-species potentials, kept so :meth:`validate_component`
            can delegate to each one's own :meth:`~Potential.validate`.
        """
        self._core = core
        self._set_units(length_unit, field_mass, G)
        self._field_pos_ho = field_pos_ho
        self._field_mass_ho = field_mass_ho
        self.components = components
        self._sub_potentials = sub_potentials

    # ------------------------------------------------------------- builders
    @classmethod
    def from_particles(
        cls,
        particles: ParticleSystem,
        components: Dict[str, ComponentSpec],
        bh_softening: float = 1e-3,
        G: float = _PotentialBase.DEFAULT_G,
        length_unit: Optional[float] = None,
    ) -> "MultiComponentPotential":
        """Fit one component per species and superpose them.

        Parameters
        ----------
        particles : ParticleSystem
            A system that has already been :meth:`~ParticleSystem.prepare`\\ d
            (recentred, aligned). Its own :attr:`~ParticleSystem.scale_radius`
            is not used -- each component estimates its own from its species
            subset.
        components : dict of str to ComponentSpec
            One :func:`scf_component`/:func:`disc_component` per species
            present in ``particles.field``. Every present species must have an
            entry, and every entry must match a present species -- there is no
            default potential type.
        bh_softening : float, optional
            Spline (Gadget4) softening length for each black hole, in
            composite length-unit units. Black holes are attached once, at
            the composite level, not per-component.
        G : float, optional
            Gravitational constant in the physical unit system (default Gadget).
        length_unit : float, optional
            Physical length used as the shared HO length unit. Defaults to the
            Hernquist-style scale radius (``r_half / (1 + sqrt(2))``, see
            :meth:`ParticleSystem.estimate_scale_radius`) of every species'
            field particles combined -- the same convention each individual
            component uses for its own scale radius, so a single-species
            system reduces exactly to that component's own result.

        Returns
        -------
        potential : MultiComponentPotential
            The superposed potential with any black holes re-attached.

        Raises
        ------
        ValueError
            If a species present in ``particles.field`` has no entry in
            ``components``, if ``components`` has an entry for a species not
            present, or if a species subset is empty.
        """
        field = particles.field
        present = set(str(s) for s in np.unique(field.species))
        specified = set(components)
        missing = present - specified
        if missing:
            raise ValueError(
                f"no ComponentSpec given for species present in the snapshot: "
                f"{sorted(missing)} (lanfear requires an explicit potential "
                f"type per species -- see lanfear.scf_component/disc_component)"
            )
        extra = specified - present
        if extra:
            raise ValueError(
                f"ComponentSpec given for species not present in the "
                f"snapshot: {sorted(extra)}"
            )

        built: Dict[str, _PotentialBase] = {}
        for label, spec in components.items():
            sub = particles.select(particles.species_mask(label))
            if sub.n_particles == 0:
                raise ValueError(f"species '{label}' has no particles")
            t0 = time.perf_counter()
            built[label] = spec.build(sub, G=G)
            elapsed = time.perf_counter() - t0
            logger.info(
                f"Built {type(built[label]).__name__} component for species "
                f"'{label}' from {sub.n_particles} particles in {elapsed:.2f} s"
            )

        field_mass = float(sum(c.field_mass for c in built.values()))
        if length_unit is None:
            # Same convention as ParticleSystem.estimate_scale_radius() (exact
            # for a Hernquist profile), computed without mutating `particles`.
            length_unit = field.half_mass_radius() / (1.0 + np.sqrt(2.0))
        length_unit = float(length_unit)

        core_components = []
        info: Dict[str, ComponentInfo] = {}
        pos_ho_parts, mass_ho_parts = [], []
        for label, comp in built.items():
            coord_scale = length_unit / comp.scale_radius
            mass_fraction = comp.field_mass / field_mass
            phi_weight = mass_fraction * coord_scale
            acc_weight = phi_weight * coord_scale
            core_components.append(
                (comp.core, coord_scale, phi_weight, acc_weight, label)
            )
            n_label = int(np.sum(field.species == label))
            info[label] = ComponentInfo(
                kind=type(comp).__name__,
                scale_radius=comp.scale_radius,
                field_mass=comp.field_mass,
                n_particles=n_label,
            )
            # Component-HO -> physical -> composite-HO, so validate() has a
            # single direct-summation reference over every species combined.
            pos_ho_parts.append(comp._field_pos_ho * comp.scale_radius / length_unit)
            mass_ho_parts.append(comp._field_mass_ho * comp.field_mass / field_mass)

        core = _core.CompositePotential(core_components)
        pos_ho = np.concatenate(pos_ho_parts)
        mass_ho = np.concatenate(mass_ho_parts)

        pot = cls(core, length_unit, field_mass, pos_ho, mass_ho, G, info, built)

        bh = particles.black_holes
        for i in range(bh.n_particles):
            pot.add_black_hole(
                mass=float(bh.mass[i]), position=bh.pos[i], softening=bh_softening
            )
        if bh.n_particles:
            logger.info(
                f"Attached {bh.n_particles} black hole(s) to the composite potential"
            )
        return pot

    # ---------------------------------------------------------- validation
    def validate(
        self,
        n_points: int = 2000,
        softening: Optional[float] = None,
        include_bh: bool = False,
        seed: int = 0,
    ) -> ValidationResult:
        """Compare the composite potential to direct summation.

        Probes a random subset of every species' field-particle positions
        combined (the general-purpose scheme, appropriate regardless of the
        composite's overall shape) and reports the fractional agreement
        against the direct-summation potential of every species together --
        the physically correct "truth", since real gravity does not know
        about the species split. To check one species' own fit in isolation,
        use :meth:`validate_component`.

        Parameters
        ----------
        n_points : int, optional
            Number of probe points sampled from the combined field particles.
        softening : float, optional
            Softening for the direct sum, in composite HO units. Defaults to
            a mean interparticle-spacing estimate.
        include_bh : bool, optional
            If True the analytic side includes the black-hole term. Left off
            by default so the check targets the field components.
        seed : int, optional
            Seed for the random probe-point selection.

        Returns
        -------
        result : ValidationResult
            The per-point and aggregate relative errors.
        """
        rng = np.random.default_rng(seed)
        n = len(self._field_pos_ho)
        idx = rng.choice(n, size=min(n_points, n), replace=False)
        pts = self._field_pos_ho[idx]

        if softening is None:
            r_hi = np.percentile(np.linalg.norm(self._field_pos_ho, axis=1), 90)
            softening = self._default_softening(r_hi)

        return self._validate_points(
            pts, np.linalg.norm(pts, axis=1), softening, include_bh
        )

    def validate_component(self, label: str, **kwargs) -> ValidationResult:
        """Goodness-of-fit of one species' own basis, in isolation.

        Delegates to that species' own fitted potential's ``validate()``,
        which compares it against direct summation of *that species alone* --
        useful for diagnosing which component is driving a poor composite fit
        (see :meth:`validate` for the combined, physically correct check).

        Parameters
        ----------
        label : str
            Species label (a key of the ``components`` dict passed to
            :meth:`from_particles`).
        **kwargs
            Passed through to that component's ``validate()`` (``Potential.validate``
            or ``DiscPotential.validate``, depending on which basis was used).

        Returns
        -------
        result : ValidationResult
            That component's own per-point/shell and aggregate relative errors.

        Raises
        ------
        KeyError
            If ``label`` is not one of this potential's components.
        """
        if label not in self._sub_potentials:
            raise KeyError(
                f"no component for species '{label}' "
                f"(have {sorted(self._sub_potentials)})"
            )
        return self._sub_potentials[label].validate(**kwargs)
