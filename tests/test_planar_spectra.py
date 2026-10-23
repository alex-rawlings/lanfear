"""Planar spectra, azimuth winding and radial frequency (classification M1/M2).

Checks the raw measurements the frame-aware classification builds on:

* the signed spectra of the planar position signals ``y + i z``, ``z + i x``
  and ``x + i y`` (``planar_lines``), whose sign gives the sense of
  circulation and which shift by exactly the pattern speed between the body
  and inertial frames;
* the unwrapped azimuth about each axis (``phi_*_net`` / ``phi_*_range``);
* the frame-independent radial frequency (``radial_frequency`` /
  ``radial_cycles``);
* the frame-aware diagnostics derived from them on ``OrbitResults``
  (winding circulation, azimuthal frequency in either frame, resolution,
  per-orbit pattern-speed ratio and pattern locking).

    python tests/test_planar_spectra.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lanfear.orbits import SUMMARY_COLUMNS, OrbitResults  # noqa: E402

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


def _results(scf, states, pattern_speed=(0.0, 0.0, 0.0), **kwargs):
    """Analyse states (HO units, time unit 1) into an OrbitResults."""
    states = np.atleast_2d(np.array(states, dtype=float))
    summary, fundamentals, lines, diffusion, planar = scf.analyse_batch(
        states,
        n_periods=N_PERIODS,
        n_samples=N_SAMPLES,
        pattern_speed=pattern_speed,
        **kwargs,
    )
    return OrbitResults(
        ids=np.arange(len(states)),
        summary=summary,
        columns=SUMMARY_COLUMNS,
        time_unit=1.0,
        length_unit=1.0,
        n_periods=N_PERIODS,
        n_samples=N_SAMPLES,
        initial_radius=np.linalg.norm(states[:, :3], axis=1),
        fundamentals=fundamentals,
        lines=lines,
        diffusion=diffusion,
        planar_lines=planar,
        pattern_speed=np.asarray(pattern_speed, dtype=float),
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


def test_winding_circulation_and_azimuthal_frequency():
    """Derived diagnostics on the pinned box and tubes (static potential)."""
    tri = build_scf(flatten=(1.0, 0.75, 0.5))
    res = _results(tri, [BOX, Z_TUBE, INNER_X_TUBE])
    winding = res.winding_circulation()
    assert winding[0].max() < 0.5, winding[0]  # box: no steady winding
    assert winding[1, 2] > 0.99 and winding[2, 0] > 0.99, winding
    asym = res.spectral_asymmetry()
    assert abs(asym[0, 2]) < 0.05  # box: mirror-symmetric
    assert asym[1, 2] > 0.3  # z-tube: lopsided towards +L_z

    frequency = res.azimuthal_frequency()
    # The inner x-tube's strongest y + i z line is a sideband (w - 2 w_x < 0);
    # the azimuthal frequency is the circulation line at the winding rate.
    planar = res.planar_lines[2, 0]
    assert planar[0, 0] < 0 < frequency[2, 0], (planar[:, 0], frequency[2, 0])
    assert np.any(np.isclose(planar[:, 0], frequency[2, 0]))
    assert np.isclose(frequency[1, 2], res.planar_lines[1, 2, 0, 0])  # z-tube
    # Static: one frame, no pattern axis, zero ratio, nothing locked.
    np.testing.assert_array_equal(res.azimuthal_frequency("inertial"), frequency)
    assert res.pattern_axis() is None
    assert not np.any(res.pattern_speed_ratio()) and not res.pattern_locked().any()

    resolution = res.frequency_resolution
    np.testing.assert_allclose(resolution, 2 * np.pi / res.column("t_total"))
    cycles = np.abs(frequency) / resolution[:, None]
    np.testing.assert_array_equal(res.resolved(frequency, k=3), cycles >= 3)
    assert res.resolved(frequency[:, 2]).shape == (3,)
    flat = res.to_dict()
    np.testing.assert_array_equal(flat["winding_circulation_z"], winding[:, 2])
    print("winding circulation / azimuthal frequency checks passed")


def test_rotating_frame_diagnostics():
    """Inertial azimuthal frequency, epsilon and pattern locking when rotating."""
    tri = build_scf(flatten=(1.0, 0.75, 0.5))
    omega = 0.05
    res = _results(
        tri, [Z_TUBE, BOX], pattern_speed=(0.0, 0.0, omega), max_body_period_factor=1
    )
    assert res.pattern_axis() == (2, omega)
    body = res.azimuthal_frequency("body")
    inertial = res.azimuthal_frequency("inertial")
    np.testing.assert_allclose(inertial[:, 2], body[:, 2] + omega)
    assert np.all(np.isnan(inertial[:, :2]))
    np.testing.assert_allclose(
        res.azimuth_net("inertial")[:, 2],
        res.azimuth_net()[:, 2] + omega * res.column("t_total"),
    )
    ratio = res.pattern_speed_ratio()
    assert np.isclose(ratio[0], omega / abs(inertial[0, 2]))
    assert not res.pattern_locked().any()
    # The box precesses steadily in the tumbling frame (Coriolis), so it winds
    # about z although its angular momentum does not circulate and it still
    # passes the centre: winding alone does not make a tube. Its precession is
    # slow compared with its radial motion, unlike a tube's circulation.
    circ_z = np.abs(res.column("Lz_mean")) / res.column("Lz_abs_mean")
    assert res.winding_circulation()[1, 2] > 0.5 and circ_z[1] < 0.5
    speed_ratio = np.abs(body[:, 2]) / res.column("radial_frequency")
    assert speed_ratio[1] < 0.3 < speed_ratio[0], speed_ratio
    holes = res.hole_fraction()
    assert holes[1, 2] < 0.1 < holes[0, 2], holes

    # Synthetic locked orbit: it sweeps -Omega t in the body frame (fixed in
    # the inertial frame) and its planar line sits at -Omega.
    t_total = 1000.0
    locked = OrbitResults(
        ids=np.arange(2),
        summary=np.zeros((2, len(SUMMARY_COLUMNS))),
        columns=SUMMARY_COLUMNS,
        time_unit=1.0,
        length_unit=1.0,
        n_periods=1,
        n_samples=2,
        initial_radius=np.ones(2),
        fundamentals=np.zeros((2, 3)),
        lines=np.zeros((2, 3, 2, 2)),
        diffusion=np.zeros((2, 3)),
        planar_lines=np.zeros((2, 3, 2, 2)),
        pattern_speed=np.array([0.0, 0.0, omega]),
    )
    col = SUMMARY_COLUMNS.index
    locked.summary[:, col("t_total")] = t_total
    rates = np.array([-omega * (1 + 1e-5), 0.5])  # locked, and a fast prograde loop
    locked.summary[:, col("phi_z_net")] = rates * t_total
    locked.summary[:, col("phi_z_range")] = np.abs(rates) * t_total
    locked.planar_lines[:, 2, 0] = np.stack([rates, np.ones(2)], axis=1)
    np.testing.assert_array_equal(locked.pattern_locked(), [True, False])
    ratio = locked.pattern_speed_ratio()
    assert ratio[0] > 1e4 and np.isclose(ratio[1], omega / (0.5 + omega))
    # An orbit swinging back and forth (net angle a tenth of its spread) has
    # no epsilon.
    locked.summary[1, col("phi_z_range")] = 10 * abs(rates[1]) * t_total
    assert np.isnan(locked.pattern_speed_ratio()[1])

    # A pattern speed far from every principal axis has no exact conversion.
    tilted = OrbitResults(
        **{**locked.__dict__, "pattern_speed": np.array([0.03, 0.0, 0.04])}
    )
    try:
        tilted.pattern_axis()
    except ValueError:
        pass
    else:
        raise AssertionError("a 37-degree tilt should be rejected")
    print("rotating-frame diagnostics checks passed")


if __name__ == "__main__":
    test_planar_sense()
    test_planar_frame_shift()
    test_radial_frequency()
    test_winding_circular()
    test_winding_circulation_and_azimuthal_frequency()
    test_rotating_frame_diagnostics()
    print("\nALL PLANAR-SPECTRA TESTS PASSED")
