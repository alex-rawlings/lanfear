"""lanfear: orbit analysis for galaxy simulations.

Reads a Gadget-4 snapshot, fits an analytical potential (Hernquist-Ostriker
SCF, a Miyamoto-Nagai disc basis, or a per-species superposition of the two
via :class:`MultiComponentPotential`, each with an arbitrary-position softened
black hole) and validates it against direct summation, integrates and
frequency-analyses orbits (MPI-parallel), then classifies them into families
(deterministically or with per-orbit posterior probabilities).

Typical use::

    import lanfear as lf

    ps = lf.ParticleSystem.from_gadget_hdf5("snap.hdf5")
    ps.prepare()                       # recentre, align, set scale radius
    pot = lf.Potential.from_particles(ps, n_max=18, l_max=7)
    result = pot.validate()
    assert result.passed(tolerance=0.02)

For a system built from physically distinct species -- e.g. a stellar disc
embedded in a dark-matter halo -- fit each species its own basis and superpose
them with :class:`MultiComponentPotential`::

    pot = lf.MultiComponentPotential.from_particles(
        ps,
        components={
            "STAR": lf.disc_component(n_radial=8, n_vert=3),
            "DM": lf.scf_component(n_max=18, l_max=7),
        },
    )
"""

from ._logging import configure as _configure, get_logger, set_verbosity
from ._package_info import print_package_info
from . import _core
from .particle_system import ParticleSystem
from .potential import Potential, TruncationSweep, ValidationResult
from .disc_potential import DiscPotential
from .multi_component_potential import (
    ComponentInfo,
    ComponentSpec,
    MultiComponentPotential,
    disc_component,
    scf_component,
)
from .orbits import (
    OrbitResults,
    SUMMARY_COLUMNS,
    analyse_family,
    analyse_states,
    ParticleTrajectory,
)
from .classify import (
    CLASS_NAMES,
    CONDENSED_NAMES,
    ClassificationComparison,
    ClassFractions,
    OrbitClass,
    OrbitClassification,
    OrbitFamily,
    classify_orbits,
)
from .probabilistic_classify import (
    BayesianOrbitClassifier,
    ProbabilisticOrbitClassification,
    classify_orbits_probabilistic,
    fit_probabilistic_classifier,
)

# Set up the package logger as soon as lanfear is imported (default WARNING).
# Control it from a calling script with lanfear.set_verbosity("INFO").
_configure()

__all__ = [
    "ParticleSystem",
    "Potential",
    "DiscPotential",
    "MultiComponentPotential",
    "ComponentSpec",
    "ComponentInfo",
    "scf_component",
    "disc_component",
    "ValidationResult",
    "TruncationSweep",
    "OrbitResults",
    "ParticleTrajectory",
    "SUMMARY_COLUMNS",
    "analyse_family",
    "analyse_states",
    "classify_orbits",
    "OrbitClass",
    "OrbitFamily",
    "OrbitClassification",
    "ClassificationComparison",
    "ClassFractions",
    "CLASS_NAMES",
    "CONDENSED_NAMES",
    "BayesianOrbitClassifier",
    "ProbabilisticOrbitClassification",
    "classify_orbits_probabilistic",
    "fit_probabilistic_classifier",
    "set_verbosity",
    "get_logger",
    "print_package_info",
    "_core",
]

__version__ = "1.2.0"
