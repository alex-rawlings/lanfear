"""Equilibrium N-body model of a triaxial perfect ellipsoid, with reference families.

Builds a self-consistent triaxial perfect ellipsoid (de Zeeuw 1985),

    rho(m) = rho_0 / (1 + m^2)^2,   m^2 = x^2/a^2 + y^2/b^2 + z^2/c^2,

with Schwarzschild's orbit-superposition method in AGAMA (Vasiliev 2019),
samples it into a Gadget-4 HDF5 initial-conditions file (one species, stars,
PartType4), and records a reference orbit family for every particle.

With ``--bh-mass-fraction`` a central black hole is added: a softened point
mass (a Plummer sphere of ``--bh-softening``) in the potential the orbits move
in, and one PartType5 particle at the centre of the initial conditions. The
stellar density is unchanged, so the stars are in equilibrium in the combined
potential. The black hole breaks the Staeckel form, so the reference families
then come from the body-frame circulation, as for a tumbling figure (below),
and AGAMA's Lyapunov exponents mark the chaotic orbits it creates.

The figure is static by default. With ``--corotation-radius`` it tumbles about
its short axis z at the pattern speed ``Omega_p = Omega_c(R_CR)`` that puts
corotation at ``R_CR`` half-mass radii (``Omega_c`` from the monopole of the
potential, as in ``make_rotation_grid.py``): the orbit library and the
reference orbits are integrated in the co-rotating frame (AGAMA's ``Omega``),
and the self-consistent model is a figure tumbling steadily at ``Omega_p``.
AGAMA's rotating-frame orbits take and return inertial velocities in the
rotating coordinates, so the velocity isotropy constraint and the initial
conditions are inertial (the frames coincide at t = 0).

The perfect ellipsoid's potential is of Staeckel form, so in a static figure
every orbit belongs to exactly one of four families, fixed by its integrals of
motion: box, short-axis tube, inner long-axis tube or outer long-axis tube. The
reference family of each library orbit is read off a long, densely sampled
integration in AGAMA's multipole expansion of the exact potential (checked
against the exact potential to ~1e-9), in the body frame if the figure
tumbles:

* In the Staeckel (confocal ellipsoidal) coordinates of the perfect ellipsoid
  (focal constants ``(-a^2, -b^2, -c^2)``, de Zeeuw 1985), each coordinate
  either reaches a boundary of its range (where the orbit crosses a symmetry
  plane) or turns back inside it. Which of the five boundaries an orbit
  reaches is its *pattern*, and each family has exactly one: on a test library
  boxes, short-axis tubes and the two long-axis-tube groups each have one
  pattern for 99.6--99.9% of their orbits. The long-axis tubes cross ``x = 0``
  either through ``t1``, i.e. inside the focal ellipse of the y-z plane (the
  inner long-axis tubes), or through ``t2``, outside it (the outer ones). The
  median waist pinch of each group is reported as a check (inner tubes are
  usually, but not always, pinched at the waist).
* As a cross-check, a tube never changes the sign of the (body-frame) angular
  momentum about its axis, a box changes all three.
* A tumbling figure is no longer integrable, and some orbits become chaotic.
  Each reference orbit's Lyapunov exponent is estimated
  (AGAMA's ``lyapunov``; in the static figure every orbit should be regular),
  and whether both halves of its trajectory show the same pattern is recorded.
* The sense of a short-axis tube is the sign of its net azimuthal winding
  about z in the inertial frame (body-frame winding plus ``Omega_p`` times the
  integration time), the quantity lanfear's prograde/retrograde label is
  based on.

* In a tumbling figure, or with a black hole, the patterns no longer separate
  the families: orbits
  cross the symmetry planes on both sides of the static potential's focal
  curves. The reference families are then defined by the body-frame
  circulation itself (a tube's angular momentum about its axis keeps its sign
  over the whole long integration and in both halves of it; a box's about no
  axis), with long-axis tubes split by where they cross ``x = 0`` (only inside
  the focal ellipse: inner; only outside: outer). This is still independent
  of lanfear's pipeline: exact potential rather than an SCF fit, a window four
  times longer and a strict sign test rather than thresholded diagnostics.

Particles inherit the family of the library orbit they were sampled from.
Orbits whose pattern is not one of the four, or contradicts the
angular-momentum check (static), or that keep the sign about the intermediate
axis or two axes or cross ``x = 0`` on both sides (tumbling), are flagged
``ambiguous``. The family of a chaotic
orbit is still its pattern's; the comparison treats chaotic orbits separately.

Run in an environment with AGAMA and h5py:

    module restore python313
    source ~/.virtualenvs/agama/bin/activate
    python scripts/make_perfect_ellipsoid.py --out-dir /orion/ptmp/arawling/orbits/agama
    python scripts/make_perfect_ellipsoid.py --corotation-radius 16 \
        --out-dir /orion/ptmp/arawling/orbits/agama/tumbling_rcr16
    python scripts/make_perfect_ellipsoid.py --bh-mass-fraction 0.01 \
        --out-dir /orion/ptmp/arawling/orbits/agama/bh_0.01

Writes ``perfect_ellipsoid_ics.hdf5`` (Gadget-4 initial conditions),
``perfect_ellipsoid_reference.npz`` (particle IDs, parent orbit, reference
family, Lyapunov exponent, sense, hole sizes and windings, library orbits and
their diagnostics) and
``perfect_ellipsoid_model.json`` (model parameters and checks).
"""

import argparse
import json
import os
import time

import h5py
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

import agama

FAMILIES = ("box", "short_axis_tube", "inner_long_axis_tube", "outer_long_axis_tube")
AMBIGUOUS = "ambiguous"


def exact_potential(points, mass, axes, G):
    """Exact potential of the perfect ellipsoid (homoeoid formula).

    For a density stratified on similar ellipsoids,
    ``Phi(x) = -pi G a b c int_0^inf [psi(inf) - psi(m^2(u))] du / Delta(u)``
    with ``psi(m^2) = int_0^{m^2} rho dm^2``; for the perfect ellipsoid this is
    ``Phi = -(G M / pi) int_0^inf du / [Delta(u) (1 + m^2(u))]``, with
    ``Delta = sqrt(prod (a_i^2 + u))`` and ``m^2(u) = sum x_i^2 / (a_i^2 + u)``.

    Parameters
    ----------
    points : numpy.ndarray
        (N, 3) positions.
    mass : float
        Total mass.
    axes : sequence of float
        (a, b, c) scale lengths along x, y, z.
    G : float
        Gravitational constant in the same units.

    Returns
    -------
    phi : numpy.ndarray
        (N,) potential.
    """
    a2 = np.asarray(axes, dtype=float) ** 2
    out = np.empty(len(points))
    for k, x in enumerate(points):
        x2 = np.asarray(x, dtype=float) ** 2

        def integrand(s):
            # u = s / (1 - s) maps [0, 1) onto [0, inf); the integrand
            # vanishes as u -> inf.
            if s >= 1.0:
                return 0.0
            u = s / (1.0 - s)
            delta = np.sqrt(np.prod(a2 + u))
            m2 = np.sum(x2 / (a2 + u))
            return 1.0 / (delta * (1.0 + m2)) / (1.0 - s) ** 2

        value, _ = quad(integrand, 0.0, 1.0, limit=200, epsabs=0, epsrel=1e-10)
        out[k] = -G * mass / np.pi * value
    return out


def ellipsoidal_coordinates(xyz, k):
    """Confocal ellipsoidal coordinates of points, for focal constants ``k``.

    The coordinates are the three roots ``t`` of ``sum x_i^2 / (t + k_i) = 1``,
    ordered ``t3 < t2 < t1``; with ``b0 < b1 < b2`` the sorted values of
    ``-k_i``, their ranges are ``b0 <= t3 <= b1 <= t2 <= b2 <= t1``.

    Parameters
    ----------
    xyz : numpy.ndarray
        (..., 3) positions.
    k : sequence of float
        (k_x, k_y, k_z). For the perfect ellipsoid with axes a > b > c these
        are ``(-a^2, -b^2, -c^2)`` (de Zeeuw 1985): the confocal ellipsoids are
        elongated along the density's short axis.

    Returns
    -------
    t1, t2, t3 : numpy.ndarray
        The coordinates, each of shape ``xyz.shape[:-1]``.
    """
    kx, ky, kz = k
    x2, y2, z2 = (np.asarray(xyz[..., i], dtype=np.float64) ** 2 for i in range(3))
    # Monic cubic t^3 + A2 t^2 + A1 t + A0 = 0.
    a2 = kx + ky + kz - (x2 + y2 + z2)
    a1 = (
        (kx * ky + kx * kz + ky * kz) - x2 * (ky + kz) - y2 * (kx + kz) - z2 * (kx + ky)
    )
    a0 = kx * ky * kz - x2 * ky * kz - y2 * kx * kz - z2 * kx * ky
    # Depressed cubic t = s - a2/3: s^3 + p s + q = 0 (three real roots).
    p = a1 - a2**2 / 3.0
    q = 2.0 * a2**3 / 27.0 - a2 * a1 / 3.0 + a0
    p = np.minimum(p, -1e-300)
    radius = 2.0 * np.sqrt(-p / 3.0)
    argument = np.clip(3.0 * q / (p * radius), -1.0, 1.0)
    phase = np.arccos(argument) / 3.0
    roots = (
        np.stack(
            [radius * np.cos(phase - 2.0 * np.pi * j / 3.0) for j in range(3)], axis=-1
        )
        - (a2 / 3.0)[..., None]
    )
    roots.sort(axis=-1)
    return roots[..., 2], roots[..., 1], roots[..., 0]


def turning_point_pattern(t1, t2, t3, k, touch_tol):
    """Which ellipsoidal-coordinate range boundaries each orbit reaches.

    Parameters
    ----------
    t1, t2, t3 : numpy.ndarray
        (n_orbits, n_samples) ellipsoidal coordinates along each orbit.
    k : sequence of float
        Focal constants (k_x, k_y, k_z).
    touch_tol : float
        A coordinate reaches a range boundary if its closest approach is
        within this fraction of its own oscillation range.

    Returns
    -------
    pattern : numpy.ndarray
        (n_orbits,) 5-character pattern (see :func:`classify_trajectories`).
    gaps : numpy.ndarray
        (n_orbits, 5) closest approach to each boundary, in units of that
        coordinate's range over the orbit.
    """
    b0, b1, b2 = np.sort(-np.asarray(k, dtype=float))

    def gap(t, bound, side):
        spread = np.maximum(np.ptp(t, axis=1), 1e-300)
        distance = (t.min(axis=1) - bound) if side == "low" else (bound - t.max(axis=1))
        return distance / spread

    gaps = np.stack(
        [
            gap(t1, b2, "low"),
            gap(t2, b1, "low"),
            gap(t2, b2, "high"),
            gap(t3, b0, "low"),
            gap(t3, b1, "high"),
        ],
        axis=1,
    )
    touches = gaps <= touch_tol
    pattern = np.array(["".join("1" if v else "0" for v in row) for row in touches])
    return pattern, gaps


def classify_trajectories(trajectories, k, touch_tol, duration, pattern_speed):
    """Tube axis, Staeckel turning-point pattern and sense of each sampled orbit.

    Parameters
    ----------
    trajectories : numpy.ndarray
        (n_orbits, n_samples, 6) positions and velocities; if the figure
        tumbles, positions in the body frame and inertial velocities in its
        coordinates (AGAMA's convention).
    k : sequence of float
        Focal constants (k_x, k_y, k_z) of the ellipsoidal coordinates.
    touch_tol : float
        A coordinate reaches a range boundary if its closest approach is
        within this fraction of its own oscillation range.
    duration : numpy.ndarray
        (n_orbits,) time spanned by each trajectory.
    pattern_speed : float
        Pattern speed of the figure about z (0 if static).

    Returns
    -------
    tube_axis : numpy.ndarray
        (n_orbits,) axis whose angular momentum keeps its sign (0, 1, 2), -1 for
        none (a box) and -2 for more than one.
    halves_agree : numpy.ndarray
        (n_orbits,) whether both halves of the trajectory give the same
        ``tube_axis`` (False flags an orbit near a separatrix).
    pattern : numpy.ndarray
        (n_orbits,) 5-character string of which coordinate range boundaries the
        orbit reaches ("1" yes, "0" no), with ``b0 < b1 < b2`` the sorted
        ``-k_i``: t1 at b2, t2 at b1, t2 at b2, t3 at b0, t3 at b1. In a
        Staeckel potential each family has its own pattern.
    pinch : numpy.ndarray
        (n_orbits,) y-z extent of the orbit near ``x = 0`` over that near its
        x-extremes (< 1: pinched at the waist), used to name the x-tube groups.
    gaps : numpy.ndarray
        (n_orbits, 5) closest approach to each boundary (same order as
        ``pattern``), in units of that coordinate's range over the orbit.
    pattern_halves_agree : numpy.ndarray
        (n_orbits,) whether both halves of the trajectory show ``pattern``.
    inertial_turns_z : numpy.ndarray
        (n_orbits,) net number of turns about z in the inertial frame over the
        trajectory (its sign is the sense of a short-axis tube).
    hole : numpy.ndarray
        (n_orbits, 3) closest approach to each axis over the largest distance
        from it (a tube avoids its axis; a box, even a precessing one, passes
        close to the centre).
    body_turns : numpy.ndarray
        (n_orbits, 3) net number of turns about each axis in the body frame.
    """
    pos = trajectories[..., :3]
    # Body-frame velocities: the tube test is about circulation in the body.
    vel = trajectories[..., 3:] - np.cross([0.0, 0.0, pattern_speed], pos)
    angular = np.cross(pos, vel)  # (n, s, 3)

    def tube_axis_of(segment):
        sign = np.sign(segment)
        constant = np.all(sign == sign[:, :1, :], axis=1) & np.all(sign != 0, axis=1)
        n_constant = constant.sum(axis=1)
        axis = np.where(n_constant == 1, np.argmax(constant, axis=1), -1)
        return np.where(n_constant > 1, -2, axis)

    half = angular.shape[1] // 2
    tube_axis = tube_axis_of(angular)
    halves_agree = (tube_axis_of(angular[:, :half]) == tube_axis) & (
        tube_axis_of(angular[:, half:]) == tube_axis
    )

    t1, t2, t3 = ellipsoidal_coordinates(pos, k)
    pattern, gaps = turning_point_pattern(t1, t2, t3, k, touch_tol)
    pattern_halves_agree = np.ones(len(pattern), dtype=bool)
    for part in (slice(None, half), slice(half, None)):
        part_pattern, _ = turning_point_pattern(
            t1[:, part], t2[:, part], t3[:, part], k, touch_tol
        )
        pattern_halves_agree &= part_pattern == pattern

    # Net inertial winding about z: the unwrapped body-frame azimuth plus the
    # frame's own rotation.
    hole = np.empty((len(pos), 3))
    body_turns = np.empty((len(pos), 3))
    for axis in range(3):
        u, v = pos[..., (axis + 1) % 3], pos[..., (axis + 2) % 3]
        distance = np.hypot(u, v)
        hole[:, axis] = distance.min(axis=1) / np.maximum(distance.max(axis=1), 1e-300)
        phi = np.unwrap(np.arctan2(v, u), axis=1)
        body_turns[:, axis] = (phi[:, -1] - phi[:, 0]) / (2.0 * np.pi)
    inertial_turns_z = body_turns[:, 2] + pattern_speed * np.asarray(duration) / (
        2.0 * np.pi
    )

    x = np.abs(pos[..., 0])
    rho_yz = np.hypot(pos[..., 1], pos[..., 2])
    x_max = x.max(axis=1, keepdims=True)
    centre = np.where(x < 0.2 * x_max, rho_yz, 0.0).max(axis=1)
    border = np.where(x > 0.8 * x_max, rho_yz, 0.0).max(axis=1)
    pinch = centre / np.maximum(border, 1e-300)
    return (
        tube_axis,
        halves_agree,
        pattern,
        pinch,
        gaps,
        pattern_halves_agree,
        inertial_turns_z,
        hole,
        body_turns,
    )


def circulation_families(tube_axis, halves_agree, pattern):
    """Reference families of a tumbling figure, from the body-frame circulation.

    A tumbling perfect ellipsoid is not integrable, and its orbits cross the
    symmetry planes on both sides of the focal curves that separate the
    families of the static potential, so the turning-point pattern no longer
    names the family. The definition of a tube remains: an orbit whose
    body-frame angular momentum about one axis keeps its sign over the whole
    (long) integration, and in both halves of it. A long-axis tube is inner if
    it crosses ``x = 0`` only inside the focal ellipse of the y-z plane (``t1``
    reaches its lower boundary, ``t2`` does not reach its upper one), outer if
    only outside it.

    Parameters
    ----------
    tube_axis, halves_agree, pattern : numpy.ndarray
        As returned by :func:`classify_trajectories` (``tube_axis`` -3 for
        unused orbits).

    Returns
    -------
    family : numpy.ndarray
        (n_orbits,) family names; ``ambiguous`` for orbits that keep the sign
        about the intermediate axis or about two axes, whose two halves
        disagree, or long-axis tubes crossing ``x = 0`` on both sides of the
        focal ellipse.
    """
    family = np.full(len(tube_axis), AMBIGUOUS, dtype=object)
    family[(tube_axis == -1) & halves_agree] = "box"
    family[(tube_axis == 2) & halves_agree] = "short_axis_tube"
    lat = (tube_axis == 0) & halves_agree
    inside = np.array([pt[:1] == "1" for pt in pattern])
    outside = np.array([pt[2:3] == "1" for pt in pattern])
    family[lat & inside & ~outside] = "inner_long_axis_tube"
    family[lat & outside & ~inside] = "outer_long_axis_tube"
    return family


def corotation_pattern_speed(den, corotation_radius):
    """Pattern speed putting corotation at ``corotation_radius`` half-mass radii.

    Parameters
    ----------
    den : agama.Density
        The model's density.
    corotation_radius : float
        Corotation radius in half-mass radii (inf for a static figure).

    Returns
    -------
    pattern_speed : float
        ``Omega_c(R_CR)`` of the monopole of the potential (0 if static).
    half_mass_radius : float
        Radius of the sphere enclosing half the mass.
    """
    total = den.totalMass()
    half_mass_radius = brentq(
        lambda r: den.enclosedMass(r) - 0.5 * total, 1e-3, 1e4, xtol=1e-10
    )
    if not np.isfinite(corotation_radius):
        return 0.0, half_mass_radius
    monopole = agama.Potential(type="Multipole", density=den, lmax=0)
    r = corotation_radius * half_mass_radius
    radial_force = monopole.force([r, 0.0, 0.0])[0]
    return float(np.sqrt(-radial_force / r)), half_mass_radius


def write_gadget_ics(path, positions, velocities, masses, ids, bh_mass=0.0):
    """Write a Gadget-4 HDF5 initial-conditions file (stars, and a black hole).

    Parameters
    ----------
    path : str
        Output file.
    positions, velocities : numpy.ndarray
        (N, 3) in kpc and km/s.
    masses : numpy.ndarray
        (N,) in 1e10 Msun.
    ids : numpy.ndarray
        (N,) particle IDs.
    bh_mass : float, optional
        Mass of a black hole at rest at the origin (PartType5, ID
        ``max(ids) + 1``), in 1e10 Msun; 0 (default) for none.
    """
    n = len(ids)
    counts = np.zeros(6, dtype=np.uint64)
    counts[4] = n
    counts[5] = 1 if bh_mass > 0 else 0
    with h5py.File(path, "w") as f:
        header = f.create_group("Header")
        header.attrs["NumPart_ThisFile"] = counts.astype(np.int64)
        header.attrs["NumPart_Total"] = counts
        header.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        header.attrs["MassTable"] = np.zeros(6)
        header.attrs["Time"] = 0.0
        header.attrs["Redshift"] = 0.0
        header.attrs["BoxSize"] = 0.0
        header.attrs["NumFilesPerSnapshot"] = 1
        stars = f.create_group("PartType4")
        stars.create_dataset("Coordinates", data=positions.astype(np.float64))
        stars.create_dataset("Velocities", data=velocities.astype(np.float64))
        stars.create_dataset("Masses", data=masses.astype(np.float64))
        stars.create_dataset("ParticleIDs", data=ids.astype(np.uint64))
        if bh_mass > 0:
            bh = f.create_group("PartType5")
            bh.create_dataset("Coordinates", data=np.zeros((1, 3)))
            bh.create_dataset("Velocities", data=np.zeros((1, 3)))
            bh.create_dataset("Masses", data=np.array([bh_mass], dtype=np.float64))
            bh.create_dataset(
                "ParticleIDs", data=np.array([ids.max() + 1], dtype=np.uint64)
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--mass", type=float, default=10.0, help="1e10 Msun (default 1e11 Msun)"
    )
    ap.add_argument("--scale-radius", type=float, default=2.0, help="a, along x (kpc)")
    ap.add_argument("--b-over-a", type=float, default=0.79)
    ap.add_argument("--c-over-a", type=float, default=0.5)
    ap.add_argument("--n-particles", type=int, default=1_000_000)
    ap.add_argument("--n-orbits", type=int, default=20_000, help="orbit library size")
    ap.add_argument("--library-periods", type=float, default=60.0)
    ap.add_argument("--reference-periods", type=float, default=200.0)
    ap.add_argument("--reference-samples", type=int, default=10_000)
    ap.add_argument("--touch-tol", type=float, default=1e-4)
    ap.add_argument(
        "--bh-mass-fraction",
        type=float,
        default=0.0,
        help="central black hole mass over the stellar mass (default 0: none)",
    )
    ap.add_argument(
        "--bh-softening",
        type=float,
        default=5e-4,
        help="Plummer softening of the black hole (kpc; default 5e-4, about "
        "lanfear's default spline softening of 1e-3 scale radii)",
    )
    ap.add_argument(
        "--corotation-radius",
        type=float,
        default=np.inf,
        help="tumble about z with corotation at this many half-mass radii "
        "(default inf: static)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    agama.setRandomSeed(args.seed)
    agama.setUnits(mass=1e10, length=1, velocity=1)  # Gadget units
    t0 = time.perf_counter()

    a = args.scale_radius
    b, c = args.b_over_a * a, args.c_over_a * a
    den = agama.Density(
        type="Spheroid",
        mass=args.mass,
        scaleRadius=a,
        axisRatioY=args.b_over_a,
        axisRatioZ=args.c_over_a,
        alpha=2,
        beta=4,
        gamma=0,
    )
    stellar_pot = agama.Potential(
        type="Multipole",
        density=den,
        lmax=16,
        mmax=16,
        gridSizeR=60,
        rmin=0.01 * a,
        rmax=1000.0 * a,
    )
    bh_mass = args.bh_mass_fraction * args.mass
    if bh_mass > 0:
        pot = agama.Potential(
            stellar_pot,
            agama.Potential(
                type="Plummer", mass=bh_mass, scaleRadius=args.bh_softening
            ),
        )
        print(
            f"central black hole: {bh_mass:.4g} (x 1e10 Msun), softening "
            f"{args.bh_softening:g} kpc",
            flush=True,
        )
    else:
        pot = stellar_pot
    # Without a black hole and rotation the potential is of Staeckel form.
    integrable = bh_mass == 0 and not np.isfinite(args.corotation_radius)
    print(f"model mass {den.totalMass():.6g} (requested {args.mass})", flush=True)
    omega, half_mass_radius = corotation_pattern_speed(den, args.corotation_radius)
    print(
        f"half-mass radius {half_mass_radius:.4g} kpc; pattern speed {omega:.6g} "
        f"km/s/kpc (corotation at {args.corotation_radius:g} half-mass radii)",
        flush=True,
    )

    # Check the multipole potential against the exact one.
    rng = np.random.default_rng(args.seed)
    test = rng.normal(size=(200, 3))
    test *= (a * np.exp(rng.uniform(np.log(0.05), np.log(50.0), 200)))[
        :, None
    ] / np.linalg.norm(test, axis=1, keepdims=True)
    exact = exact_potential(test, args.mass, (a, b, c), agama.G)
    rel_err = np.abs(stellar_pot.potential(test) / exact - 1.0)
    print(
        f"potential vs exact: median rel. error {np.median(rel_err):.2e}, max {rel_err.max():.2e}",
        flush=True,
    )

    # Schwarzschild orbit library: density matched, velocities isotropic (in
    # the inertial frame: AGAMA's velocities are inertial even in a rotating
    # frame).
    ics, _ = den.sample(args.n_orbits, potential=pot)
    periods = pot.Tcirc(ics)
    target_density = agama.Target(
        type="DensityClassicLinear",
        gridr=agama.nonuniformGrid(30, 0.05 * a, 100.0 * a),
        axisratioy=args.b_over_a,
        axisratioz=args.c_over_a,
        stripsPerPane=2,
    )
    target_kinematics = agama.Target(
        type="KinemShell", gridR=agama.nonuniformGrid(20, 0.1 * a, 50.0 * a), degree=1
    )
    data_density, data_kinematics, trajectories = agama.orbit(
        potential=pot,
        ic=ics,
        time=args.library_periods * periods,
        dtype=object,
        targets=[target_density, target_kinematics],
        Omega=omega,
    )
    print(f"library integrated ({time.perf_counter() - t0:.0f} s)", flush=True)
    n_grid = len(target_kinematics) // 2
    isotropy = 2 * data_kinematics[:, :n_grid] - data_kinematics[:, n_grid:]
    rhs_density = target_density(den)
    weights = agama.solveOpt(
        matrix=(np.ones((len(ics), 1)).T, data_density.T, isotropy.T),
        rhs=([den.totalMass()], rhs_density, np.zeros(n_grid)),
        rpenl=([np.inf], np.ones_like(rhs_density), np.ones_like(np.zeros(n_grid))),
        xpenq=np.ones(len(ics)) * 0.1,
    )
    density_error = np.abs(data_density.T.dot(weights) / rhs_density - 1.0)
    print(
        f"orbit weights solved: {np.sum(weights > 0)} of {len(weights)} orbits used; "
        f"density constraints max rel. error {density_error.max():.2e}",
        flush=True,
    )

    # N-body sample. In a static figure, flip a random half of the velocities so
    # the model does not rotate (v -> -v maps an equilibrium orbit onto another
    # one of the same family). In a tumbling one time reversal is not a symmetry
    # (it reverses the Coriolis force), so the streaming is the solution's own.
    phase_space, masses, parent = agama.sampleOrbitLibrary(
        args.n_particles, trajectories, weights, returnIndices=True
    )
    if omega == 0.0:
        flip = np.random.random(len(masses)) < 0.5
        phase_space[flip, 3:] *= -1.0
    ids = np.arange(1, len(masses) + 1, dtype=np.uint64)
    ics_path = os.path.join(args.out_dir, "perfect_ellipsoid_ics.hdf5")
    write_gadget_ics(
        ics_path, phase_space[:, :3], phase_space[:, 3:], masses, ids, bh_mass
    )
    l_each = masses[:, None] * np.cross(phase_space[:, :3], phase_space[:, 3:])
    net_l = l_each.sum(axis=0)
    l_scale = np.abs(l_each).sum(axis=0)
    print(
        f"wrote {ics_path}: {len(ids)} particles; net L / sum |L| = "
        f"{np.round(net_l / l_scale, 5)} (shot noise ~{1 / np.sqrt(len(ids)):.1e})",
        flush=True,
    )

    # Reference families of the library orbits, from long dense integrations
    # (in the body frame), processed in chunks to bound the memory.
    used = np.flatnonzero(weights > 0)
    tube_axis = np.full(len(ics), -3, dtype=np.int64)  # -3: unused orbit
    agree = np.zeros(len(ics), dtype=bool)
    pattern = np.full(len(ics), "", dtype="<U5")
    pinch = np.full(len(ics), np.nan)
    gaps = np.full((len(ics), 5), np.nan)
    pattern_agree = np.zeros(len(ics), dtype=bool)
    inertial_turns = np.full(len(ics), np.nan)
    hole = np.full((len(ics), 3), np.nan)
    body_turns = np.full((len(ics), 3), np.nan)
    lyapunov = np.full((len(ics), 2), np.nan)  # (lambda T_circ, T_chaos / T_circ)
    chunk = 1000
    for start in range(0, len(used), chunk):
        sel = used[start : start + chunk]
        duration = args.reference_periods * periods[sel]
        _, traj, lyapunov[sel] = agama.orbit(
            potential=pot,
            ic=ics[sel],
            time=duration,
            trajsize=args.reference_samples,
            dtype="float64",
            separateTime=True,
            lyapunov=True,
            Omega=omega,
            verbose=False,
        )
        (
            tube_axis[sel],
            agree[sel],
            pattern[sel],
            pinch[sel],
            gaps[sel],
            pattern_agree[sel],
            inertial_turns[sel],
            hole[sel],
            body_turns[sel],
        ) = classify_trajectories(
            np.stack(traj), (-a * a, -b * b, -c * c), args.touch_tol, duration, omega
        )
        print(
            f"  reference: {min(start + chunk, len(used))} of {len(used)} orbits",
            flush=True,
        )

    # Turning-point patterns per tube axis (a check that the Staeckel focal
    # constants are right: each family should have one dominant pattern).
    for axis, name in (
        (-1, "box"),
        (0, "x-tube"),
        (1, "y-tube"),
        (2, "z-tube"),
        (-2, "multi"),
    ):
        sel = (tube_axis == axis) & agree
        if sel.any():
            values, counts = np.unique(pattern[sel], return_counts=True)
            order = np.argsort(-counts)
            top = ", ".join(f"{values[i]}: {counts[i]}" for i in order[:4])
            print(f"  patterns of {name}s ({sel.sum()}): {top}", flush=True)

    if integrable:
        # Families by Staeckel pattern: the dominant pattern of the boxes and of the
        # z-tubes (angular-momentum check), and the two most common x-tube
        # patterns: the one whose t1 reaches its lower boundary crosses x = 0 inside
        # the focal ellipse (inner), the other through t2, outside it (outer).
        family_of_pattern = {}
        for axis, name in ((-1, "box"), (2, "short_axis_tube")):
            sel = (tube_axis == axis) & agree
            values, counts = np.unique(pattern[sel], return_counts=True)
            family_of_pattern[str(values[np.argmax(counts)])] = name
        x_tubes = (tube_axis == 0) & agree
        values, counts = np.unique(pattern[x_tubes], return_counts=True)
        x_groups = {}
        for i in np.argsort(-counts)[:2]:
            median = float(np.median(pinch[x_tubes & (pattern == values[i])]))
            name = (
                "inner_long_axis_tube"
                if values[i][0] == "1"
                else "outer_long_axis_tube"
            )
            family_of_pattern[str(values[i])] = name
            x_groups[str(values[i])] = {
                "count": int(counts[i]),
                "median_pinch": median,
                "family": name,
            }
        if len(set(family_of_pattern.values())) != 4:
            print(
                f"WARNING: patterns do not map onto four families: {family_of_pattern}",
                flush=True,
            )
        print(
            f"family of each pattern: {family_of_pattern}; x-tube groups: {x_groups}",
            flush=True,
        )

        family = np.array(
            [family_of_pattern.get(str(pt), AMBIGUOUS) for pt in pattern], dtype=object
        )
        # Cross-check with the angular momentum: an orbit whose single sign-keeping
        # axis contradicts its pattern's family is ambiguous (orbits keeping the
        # sign about two axes, i.e. thin ones, are decided by the pattern alone).
        expected_axis = {
            "box": -1,
            "short_axis_tube": 2,
            "inner_long_axis_tube": 0,
            "outer_long_axis_tube": 0,
        }
        for name, axis in expected_axis.items():
            family[(family == name) & (tube_axis != axis) & (tube_axis != -2)] = (
                AMBIGUOUS
            )
    else:
        family = circulation_families(tube_axis, agree, pattern)
        family_of_pattern = x_groups = None
        print(
            "families from the body-frame circulation (the Staeckel patterns only "
            "split the long-axis tubes)",
            flush=True,
        )
    family[tube_axis == -3] = "unused"
    w = weights / weights.sum()
    mass_fractions = {f: float(w[family == f].sum()) for f in (*FAMILIES, AMBIGUOUS)}
    print(f"reference mass fractions: {mass_fractions}", flush=True)
    # AGAMA reports lambda = 0 without evidence of chaos, NaN if undecided.
    chaotic = lyapunov[:, 0] > 0
    undecided = np.isnan(lyapunov[:, 0]) & (tube_axis != -3)
    print(
        f"chaotic (Lyapunov) mass fraction {w[chaotic].sum():.4f}"
        f" ({int(chaotic.sum())} orbits; {int(undecided.sum())} undecided); "
        f"pattern differs between trajectory halves for mass fraction "
        f"{w[~pattern_agree & (tube_axis != -3)].sum():.4f}",
        flush=True,
    )
    for name in FAMILIES:
        sel = family == name
        if sel.any():
            print(
                f"  {name}: chaotic mass fraction {w[sel & chaotic].sum() / w[sel].sum():.4f}",
                flush=True,
            )
    sat = family == "short_axis_tube"
    if omega != 0.0 and sat.any():
        print(
            f"short-axis tubes prograde in the inertial frame: mass fraction "
            f"{w[sat & (inertial_turns > 0)].sum() / w[sat].sum():.4f}",
            flush=True,
        )

    np.savez(
        os.path.join(args.out_dir, "perfect_ellipsoid_reference.npz"),
        particle_ids=ids,
        parent_orbit=parent.astype(np.int64),
        particle_family=family[parent].astype(str),
        orbit_ics=ics,
        orbit_weights=weights,
        orbit_periods=periods,
        orbit_family=family.astype(str),
        orbit_tube_axis=tube_axis,
        orbit_halves_agree=agree,
        orbit_pattern=pattern,
        orbit_pinch=pinch,
        orbit_gaps=gaps,
        orbit_pattern_halves_agree=pattern_agree,
        orbit_inertial_turns_z=inertial_turns,
        orbit_lyapunov=lyapunov,
        orbit_hole=hole,
        orbit_body_turns=body_turns,
        particle_lyapunov=lyapunov[parent, 0],
        particle_inertial_turns_z=inertial_turns[parent],
        particle_pattern_halves_agree=pattern_agree[parent],
        particle_hole=hole[parent],
        particle_body_turns=body_turns[parent],
    )
    with open(os.path.join(args.out_dir, "perfect_ellipsoid_model.json"), "w") as f:
        json.dump(
            {
                **vars(args),
                "axes_kpc": [a, b, c],
                "corotation_radius_half_mass": (
                    None
                    if not np.isfinite(args.corotation_radius)
                    else args.corotation_radius
                ),
                "half_mass_radius_kpc": half_mass_radius,
                "pattern_speed": omega,
                "bh_mass": bh_mass,
                "bh_softening_kpc": args.bh_softening if bh_mass > 0 else None,
                "units": "Gadget: kpc, km/s, 1e10 Msun",
                "G": agama.G,
                "potential_rel_error_median": float(np.median(rel_err)),
                "potential_rel_error_max": float(rel_err.max()),
                "density_constraint_max_rel_error": float(density_error.max()),
                "net_angular_momentum": net_l.tolist(),
                "net_angular_momentum_over_sum_abs": (net_l / l_scale).tolist(),
                "x_tube_groups": x_groups,
                "reference_method": "staeckel_pattern"
                if integrable
                else "body_frame_circulation",
                "family_of_pattern": family_of_pattern,
                "reference_mass_fractions": mass_fractions,
                "chaotic_mass_fraction": float(w[chaotic].sum()),
            },
            f,
            indent=2,
        )
    print(f"done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
