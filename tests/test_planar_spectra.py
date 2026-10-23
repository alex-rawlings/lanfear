"""Planar spectra, azimuth winding and radial frequency (classification M1).

Checks the raw measurements the frame-aware classification builds on:

* the signed spectra of the planar position signals ``y + i z``, ``z + i x``
  and ``x + i y`` (``planar_lines``), whose sign gives the sense of
  circulation and which shift by exactly the pattern speed between the body
  and inertial frames;
* the unwrapped azimuth about each axis (``phi_*_net`` / ``phi_*_range``);
* the frame-independent radial frequency (``radial_frequency`` /
  ``radial_cycles``).

    python tests/test_planar_spectra.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lanfear.orbits import SUMMARY_COLUMNS  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_classify import build_scf  # noqa: E402

N_PERIODS = 40
N_SAMPLES = 4096

# Pinned orbits of test_classify.test_known_orbits, in the (1, 0.75, 0.5) model.
BOX = [1.5, 1.0, 0.7, 0.0, 0.0, 0.0]
Z_TUBE = [2.0, 0.0, 0.6, 0.0, 0.6, 0.15]
INNER_X_TUBE = [-3.011, 0.766, 3.396, -0.192, -0.035, 0.203]


def _column(summary, name):
    return summary[SUMMARY_COLUMNS.index(name)]


def _analyse(scf, state, **kwargs):
    return scf.analyse_orbit(
        np.array(state, dtype=float),
        n_periods=N_PERIODS,
        n_samples=N_SAMPLES,
        **kwargs,
    )


def _signed_extremes(lines):
    """Strongest positive- and negative-frequency amplitude of one plane."""
    freq, amp = lines[:, 0], lines[:, 1]
    return amp[freq > 0].max(initial=0.0), amp[freq < 0].max(initial=0.0)


def test_planar_sense():
    """The planar spectrum and the winding give the sense of circulation."""
    tri = build_scf(flatten=(1.0, 0.75, 0.5))
    for sign in (1.0, -1.0):
        state = np.array(Z_TUBE)
        state[3:] *= sign
        summary, _, _, _, planar = _analyse(tri, state)
        lz = _column(summary, "Lz_mean")
        assert np.sign(lz) == sign
        # z plane: the leading line has the sign of L_z and dominates its mirror.
        assert np.sign(planar[2, 0, 0]) == sign, planar[2]
        plus, minus = _signed_extremes(planar[2])
        assert (plus > 2 * minus) if sign > 0 else (minus > 2 * plus)
        # Winding: monotonic circulation about z, at the leading line's rate.
        net = _column(summary, "phi_z_net")
        assert np.sign(net) == sign
        assert np.isclose(abs(net), _column(summary, "phi_z_range"))
        rate = net / _column(summary, "t_total")
        assert abs(rate / planar[2, 0, 0] - 1) < 0.01, (rate, planar[2, 0, 0])

    # A box has no net circulation: mirror-symmetric z-plane lines, ~no winding.
    summary, _, _, _, planar = _analyse(tri, BOX)
    plus, minus = _signed_extremes(planar[2])
    assert abs(plus - minus) < 0.05 * (plus + minus), planar[2]
    net, spread = _column(summary, "phi_z_net"), _column(summary, "phi_z_range")
    assert abs(net) < 0.1 * spread, (net, spread)

    # A long-axis tube winds steadily about x. Its y + i z spectrum carries
    # strong sidebands (the tube's cross-section changes with x), so the
    # circulation frequency is the line at the winding rate, not necessarily
    # the strongest line.
    summary, _, _, _, planar = _analyse(tri, INNER_X_TUBE)
    net = _column(summary, "phi_x_net")
    assert np.sign(net) == np.sign(_column(summary, "Lx_mean"))
    assert np.isclose(abs(net), _column(summary, "phi_x_range"))
    rate = net / _column(summary, "t_total")
    closest = planar[0, np.argmin(np.abs(planar[0, :, 0] - rate)), 0]
    assert abs(closest / rate - 1) < 0.01, (rate, planar[0])
    print("planar sense and winding checks passed")


def test_planar_frame_shift():
    """In the plane perpendicular to Omega_p, frames differ by exactly Omega_p.

    The body-frame planar signal x_b + i y_b of an orbit in a figure tumbling
    about z is analysed by the core; the inertial-frame signal
    ``(x_b + i y_b) exp(i Omega t)`` is rebuilt here from the same trajectory
    and its leading line located by maximising the Hann-windowed projection.
    The two must differ by Omega to NAFF precision.
    """
    tri = build_scf(flatten=(1.0, 0.75, 0.5))
    omega = 0.05
    pattern = (0.0, 0.0, omega)
    summary, _, _, _, planar = _analyse(
        tri, Z_TUBE, pattern_speed=pattern, max_body_period_factor=1
    )
    body = planar[2, 0, 0]

    summ_traj, traj = tri.integrate_orbit(
        np.array(Z_TUBE, dtype=float),
        N_PERIODS,
        N_SAMPLES,
        return_trajectory=True,
        pattern_speed=pattern,
        max_body_period_factor=1,
    )
    assert len(traj) == N_SAMPLES
    t = np.arange(N_SAMPLES) * _column(summ_traj, "t_total") / (N_SAMPLES - 1)
    signal = (traj[:, 0] + 1j * traj[:, 1]) * np.exp(1j * omega * t)
    signal -= signal.mean()
    window = np.hanning(N_SAMPLES)
    bin_width = 2 * np.pi / t[-1]
    grid = body + omega + np.linspace(-bin_width, bin_width, 4001)
    power = np.abs(np.exp(-1j * np.outer(grid, t)) @ (window * signal))
    inertial = grid[np.argmax(power)]
    step = grid[1] - grid[0]
    assert abs(inertial - (body + omega)) <= step, (inertial, body + omega)
    print(
        f"frame shift OK: body {body:.5f} + Omega {omega} = {body + omega:.5f}, "
        f"inertial {inertial:.5f}"
    )


def test_radial_frequency():
    """The radial frequency matches the pericentre rate and is frame-blind."""
    sph = build_scf(flatten=(1.0, 1.0, 1.0), l_max=0)
    r0 = 1.0
    vc = np.sqrt(-sph.acceleration(r0, 0, 0)[0] * r0)
    state = [r0, 0.0, 0.0, 0.0, 0.5 * vc, 0.0]
    summary, _, _, _, _ = _analyse(sph, state)
    w_r = _column(summary, "radial_frequency")
    t_total = _column(summary, "t_total")
    assert np.isclose(_column(summary, "radial_cycles"), w_r * t_total / (2 * np.pi))

    # Pericentre passages counted from the sampled radius.
    traj_summary, traj = sph.integrate_orbit(
        np.array(state), N_PERIODS, N_SAMPLES, return_trajectory=True
    )
    assert np.isnan(_column(traj_summary, "radial_frequency"))  # analysis only
    r = np.linalg.norm(traj[:, :3], axis=1)
    n_peri = np.sum((r[1:-1] < r[:-2]) & (r[1:-1] < r[2:]))
    assert abs(w_r / (2 * np.pi * n_peri / t_total) - 1) < 0.02, (w_r, n_peri)

    # A near-circular orbit's tiny epicycle gives the epicyclic frequency
    # kappa = sqrt(0.5) of a Hernquist sphere at r = a (HO units).
    circular, _, _, _, _ = _analyse(sph, [r0, 0.0, 0.0, 0.0, vc, 0.0])
    assert abs(_column(circular, "radial_frequency") / np.sqrt(0.5) - 1) < 0.01

    # r and v_r do not depend on the frame: a spherical potential rotating
    # about z gives the same radial frequency.
    rotating, _, _, _, _ = _analyse(
        sph, state, pattern_speed=(0.0, 0.0, 0.3), max_body_period_factor=1
    )
    assert abs(_column(rotating, "radial_frequency") / w_r - 1) < 1e-4
    print(f"radial frequency OK: {w_r:.4f} ({n_peri} pericentres)")


def test_winding_circular():
    """A circular orbit sweeps 2 pi per circular period about its spin axis."""
    sph = build_scf(flatten=(1.0, 1.0, 1.0), l_max=0)
    r0 = 1.0
    vc = np.sqrt(-sph.acceleration(r0, 0, 0)[0] * r0)
    summary, _, _, _, planar = _analyse(sph, [r0, 0.0, 0.0, 0.0, vc, 0.0])
    net = _column(summary, "phi_z_net")
    assert abs(net / (2 * np.pi * N_PERIODS) - 1) < 0.01, net / (2 * np.pi)
    assert np.isclose(net, _column(summary, "phi_z_range"))
    assert planar[2, 0, 0] > 0
    print(f"circular winding OK: {net / (2 * np.pi):.2f} circuits")


if __name__ == "__main__":
    test_planar_sense()
    test_planar_frame_shift()
    test_radial_frequency()
    test_winding_circular()
    print("\nALL PLANAR-SPECTRA TESTS PASSED")
