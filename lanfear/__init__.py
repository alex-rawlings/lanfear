"""lanfear: orbit analysis for galaxy simulations.

Milestone 1 provides snapshot loading, the Hernquist-Ostriker SCF potential
(with an arbitrary-position softened black hole), and validation of the
analytical potential against direct summation.

Typical use::

    import lanfear as lf

    ps = lf.ParticleSystem.from_gadget_hdf5("snap.hdf5")
    ps.prepare()                       # recentre, align, set scale radius
    pot = lf.Potential.from_particles(ps, n_max=18, l_max=7)
    result = pot.validate()
    assert result.passed(tolerance=0.02)
"""

from ._logging import configure as _configure, get_logger, set_verbosity
from . import _core
from .particle_system import ParticleSystem
from .potential import Potential, ValidationResult
from .disc_potential import DiscPotential
from .orbits import (
    OrbitResults,
    SUMMARY_COLUMNS,
    analyse_family,
    analyse_states,
    integrate_family,
    integrate_states,
)
from .classify import (
    CLASS_NAMES,
    CONDENSED_NAMES,
    OrbitClass,
    OrbitClassification,
    OrbitFamily,
    classify_orbits,
)

# Set up the package logger as soon as lanfear is imported (default WARNING).
# Control it from a calling script with lanfear.set_verbosity("INFO").
_configure()

__all__ = [
    "ParticleSystem",
    "Potential",
    "DiscPotential",
    "ValidationResult",
    "OrbitResults",
    "SUMMARY_COLUMNS",
    "integrate_family",
    "integrate_states",
    "analyse_family",
    "analyse_states",
    "classify_orbits",
    "OrbitClass",
    "OrbitFamily",
    "OrbitClassification",
    "CLASS_NAMES",
    "CONDENSED_NAMES",
    "set_verbosity",
    "get_logger",
    "_core",
]

__version__ = "0.5.0"
