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
    OrbitClass,
    OrbitClassification,
    classify_orbits,
)

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
    "OrbitClassification",
    "CLASS_NAMES",
    "_core",
]

__version__ = "0.5.0"
