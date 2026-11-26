"""SCF (Hernquist-Ostriker) potential tests.

Checks that the analytic acceleration of the spheroidal SCF expansion matches a
finite-difference gradient of its potential, in particular at exactly the
scale radius (r = 1 in HO units). There xi = (r - 1)/(r + 1) = 0, where every
odd-n Gegenbauer polynomial C_n^{alpha}(xi) vanishes exactly; a radial
derivative written as Phi_tilde * C_{n-1}^{alpha+1}/C_n^{alpha} would drop
those terms.

    python tests/test_scf_potential.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lanfear import _core  # noqa: E402


def build_flattened_hernquist_scf(n=20_000, n_max=10, l_max=4, seed=1):
    """Triaxially squashed unit Hernquist sphere, so many (n, l, m) are non-zero."""
    rng = np.random.default_rng(seed)
    su = np.sqrt(rng.uniform(0, 1, n))
    r = su / (1 - su)
    mu = rng.uniform(-1, 1, n)
    az = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - mu**2)
    pos = np.stack(
        [r * st * np.cos(az), 0.8 * r * st * np.sin(az), 0.6 * r * mu], axis=1
    )
    return _core.SCFPotential(n_max, l_max, pos, np.full(n, 1.0 / n))


def finite_difference_acceleration(pot, point, step=1e-5):
    """Central-difference -grad(Phi) at ``point``."""
    grad = np.empty(3)
    for i, e in enumerate(np.eye(3)):
        grad[i] = (
            pot.potential(*(point + step * e)) - pot.potential(*(point - step * e))
        ) / (2 * step)
    return -grad


def test_acceleration_matches_gradient_at_scale_radius():
    """Analytic acceleration == -grad(Phi) at and either side of r = 1."""
    pot = build_flattened_hernquist_scf()
    direction = np.array([0.3, 0.5, 0.8])
    direction /= np.linalg.norm(direction)
    for radius in [1.0 - 1e-6, 1.0, 1.0 + 1e-6, 0.5, 2.0]:
        point = radius * direction
        acc = np.array(pot.acceleration(*point))
        acc_fd = finite_difference_acceleration(pot, point)
        rel_err = np.linalg.norm(acc - acc_fd) / np.linalg.norm(acc_fd)
        print(f"r = {radius:.7f}: relative error = {rel_err:.3e}")
        assert rel_err < 1e-7, f"acceleration mismatch at r = {radius}: {rel_err:.3e}"


if __name__ == "__main__":
    test_acceleration_matches_gradient_at_scale_radius()
    print("\nALL SCF POTENTIAL TESTS PASSED")
