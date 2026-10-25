"""Equilibrium N-body model of a triaxial perfect ellipsoid, with reference families.

Builds a self-consistent, non-rotating triaxial perfect ellipsoid (de Zeeuw
1985),

    rho(m) = rho_0 / (1 + m^2)^2,   m^2 = x^2/a^2 + y^2/b^2 + z^2/c^2,

with Schwarzschild's orbit-superposition method in AGAMA (Vasiliev 2019),
samples it into a Gadget-4 HDF5 initial-conditions file (one species, stars,
PartType4), and records a reference orbit family for every particle.

The perfect ellipsoid's potential is of Staeckel form, so every orbit belongs
to exactly one of four families, fixed by its integrals of motion: box,
short-axis tube, inner long-axis tube or outer long-axis tube. The reference
family of each library orbit is read off a long, densely sampled integration
in AGAMA's multipole expansion of the exact potential (checked against the
exact potential to ~1e-9):

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
* As a cross-check, a tube never changes the sign of the angular momentum
  about its axis, a box changes all three.

Particles inherit the family of the library orbit they were sampled from.
Orbits whose pattern is not one of the four, or contradicts the
angular-momentum check, are flagged ``ambiguous``.

Run in an environment with AGAMA and h5py:

    module restore python313
    source ~/.virtualenvs/agama/bin/activate
    python scripts/make_perfect_ellipsoid.py --out-dir /orion/ptmp/arawling/orbits/agama

Writes ``perfect_ellipsoid_ics.hdf5`` (Gadget-4 initial conditions),
``perfect_ellipsoid_reference.npz`` (particle IDs, parent orbit, reference
family, library orbits and their diagnostics) and
``perfect_ellipsoid_model.json`` (model parameters and checks).
"""

import argparse
import json
import os
import time

import h5py
import numpy as np
from scipy.integrate import quad

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
            # u = s / (1 - s) maps [0, 1) onto [0, inf).
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


def classify_trajectories(trajectories, k, touch_tol):
    """Tube axis and Staeckel turning-point pattern of each sampled orbit.

    Parameters
    ----------
    trajectories : numpy.ndarray
        (n_orbits, n_samples, 6) positions and velocities.
    k : sequence of float
        Focal constants (k_x, k_y, k_z) of the ellipsoidal coordinates.
    touch_tol : float
        A coordinate reaches a range boundary if its closest approach is
        within this fraction of its own oscillation range.

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
    """
    pos, vel = trajectories[..., :3], trajectories[..., 3:]
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

    b0, b1, b2 = np.sort(-np.asarray(k, dtype=float))
    t1, t2, t3 = ellipsoidal_coordinates(pos, k)

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

    x = np.abs(pos[..., 0])
    rho_yz = np.hypot(pos[..., 1], pos[..., 2])
    x_max = x.max(axis=1, keepdims=True)
    centre = np.where(x < 0.2 * x_max, rho_yz, 0.0).max(axis=1)
    border = np.where(x > 0.8 * x_max, rho_yz, 0.0).max(axis=1)
    pinch = centre / np.maximum(border, 1e-300)
    return tube_axis, halves_agree, pattern, pinch, gaps


def write_gadget_ics(path, positions, velocities, masses, ids):
    """Write a Gadget-4 HDF5 initial-conditions file (PartType4 only).

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
    """
    n = len(ids)
    counts = np.zeros(6, dtype=np.uint64)
    counts[4] = n
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
    pot = agama.Potential(
        type="Multipole",
        density=den,
        lmax=16,
        mmax=16,
        gridSizeR=60,
        rmin=0.01 * a,
        rmax=1000.0 * a,
    )
    print(f"model mass {den.totalMass():.6g} (requested {args.mass})", flush=True)

    # Check the multipole potential against the exact one.
    rng = np.random.default_rng(args.seed)
    test = rng.normal(size=(200, 3))
    test *= (a * np.exp(rng.uniform(np.log(0.05), np.log(50.0), 200)))[
        :, None
    ] / np.linalg.norm(test, axis=1, keepdims=True)
    exact = exact_potential(test, args.mass, (a, b, c), agama.G)
    rel_err = np.abs(pot.potential(test) / exact - 1.0)
    print(
        f"potential vs exact: median rel. error {np.median(rel_err):.2e}, max {rel_err.max():.2e}",
        flush=True,
    )

    # Schwarzschild orbit library: density matched, velocities isotropic.
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

    # N-body sample; flip a random half of the velocities so the model does not
    # rotate (v -> -v maps an equilibrium orbit onto another one of the same family).
    phase_space, masses, parent = agama.sampleOrbitLibrary(
        args.n_particles, trajectories, weights, returnIndices=True
    )
    flip = np.random.random(len(masses)) < 0.5
    phase_space[flip, 3:] *= -1.0
    ids = np.arange(1, len(masses) + 1, dtype=np.uint64)
    ics_path = os.path.join(args.out_dir, "perfect_ellipsoid_ics.hdf5")
    write_gadget_ics(ics_path, phase_space[:, :3], phase_space[:, 3:], masses, ids)
    l_each = masses[:, None] * np.cross(phase_space[:, :3], phase_space[:, 3:])
    net_l = l_each.sum(axis=0)
    l_scale = np.abs(l_each).sum(axis=0)
    print(
        f"wrote {ics_path}: {len(ids)} particles; net L / sum |L| = "
        f"{np.round(net_l / l_scale, 5)} (shot noise ~{1 / np.sqrt(len(ids)):.1e})",
        flush=True,
    )

    # Reference families of the library orbits, from long dense integrations,
    # processed in chunks to bound the memory.
    used = np.flatnonzero(weights > 0)
    tube_axis = np.full(len(ics), -3, dtype=np.int64)  # -3: unused orbit
    agree = np.zeros(len(ics), dtype=bool)
    pattern = np.full(len(ics), "", dtype="<U5")
    pinch = np.full(len(ics), np.nan)
    gaps = np.full((len(ics), 5), np.nan)
    chunk = 1000
    for start in range(0, len(used), chunk):
        sel = used[start : start + chunk]
        out = agama.orbit(
            potential=pot,
            ic=ics[sel],
            time=args.reference_periods * periods[sel],
            trajsize=args.reference_samples,
            dtype="float64",
            separateTime=False,
        )
        traj = np.stack(out[:, 1])
        tube_axis[sel], agree[sel], pattern[sel], pinch[sel], gaps[sel] = (
            classify_trajectories(traj, (-a * a, -b * b, -c * c), args.touch_tol)
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
        name = "inner_long_axis_tube" if values[i][0] == "1" else "outer_long_axis_tube"
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
        family[(family == name) & (tube_axis != axis) & (tube_axis != -2)] = AMBIGUOUS
    family[tube_axis == -3] = "unused"
    w = weights / weights.sum()
    mass_fractions = {f: float(w[family == f].sum()) for f in (*FAMILIES, AMBIGUOUS)}
    print(f"reference mass fractions: {mass_fractions}", flush=True)

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
    )
    with open(os.path.join(args.out_dir, "perfect_ellipsoid_model.json"), "w") as f:
        json.dump(
            {
                **vars(args),
                "axes_kpc": [a, b, c],
                "units": "Gadget: kpc, km/s, 1e10 Msun",
                "G": agama.G,
                "potential_rel_error_median": float(np.median(rel_err)),
                "potential_rel_error_max": float(rel_err.max()),
                "density_constraint_max_rel_error": float(density_error.max()),
                "net_angular_momentum": net_l.tolist(),
                "net_angular_momentum_over_sum_abs": (net_l / l_scale).tolist(),
                "x_tube_groups": x_groups,
                "family_of_pattern": family_of_pattern,
                "reference_mass_fractions": mass_fractions,
            },
            f,
            indent=2,
        )
    print(f"done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
