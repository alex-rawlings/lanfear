"""Integrate a grid of rotating figures with known pattern speeds.

Ground-truth data for calibrating the classification of rotating figures
(``AI_output/classification_milestones.md``, M0 and M10). Each model is a
Hernquist profile stratified on similar ellipsoids (Valluri & Merritt 1998,
gamma = 1) with isotropic velocities from the spherical-equivalent Hernquist
distribution function, drawn in the frame co-rotating with the figure. The
figure tumbles about its short axis at the pattern speed that puts corotation
at a chosen radius, ``Omega_p = Omega_c(R_CR)`` of the spherical-equivalent
Hernquist model, with ``R_CR`` given in half-mass radii. The potential rotates
at exactly that (known) pattern speed rather than at an estimate; the
single-snapshot estimate and its uncertainty are recorded alongside for
comparison.

Two models are built:

* ``triaxial``: ``b/a = 0.79``, ``c/a = 0.5`` (triaxiality 0.5; Valluri &
  Merritt 1998), a slow tumbler such as an elliptical or merger remnant;
* ``bar``: ``b/a = 0.45``, ``c/a = 0.35``, a crude bar-like (strongly
  elongated) figure for the fast-rotator regime. It is Hernquist-stratified,
  not a Ferrers bar, so treat its bar-family content as indicative only.

Every run integrates the same random subset of particles of a model (same
seed), so labels can be compared run by run. Results are written to
``--out-dir`` as ``rotation_grid_<model>_<rcr>_...npz`` (``OrbitResults``)
with a ``.json`` sidecar of the true and estimated pattern speeds; runs already
saved are skipped. Under MPI (``srun``/``mpirun``) the orbits are spread over
the ranks; the model is built on the root rank, and if that fails every rank
exits with an error (see ``run_orbits_mpi.run_on_root``).

    python scripts/make_rotation_grid.py --out-dir test_scripts/data/rotation_grid
    srun -n 8 python scripts/make_rotation_grid.py --out-dir DIR --models triaxial

Run it after the summary layout of milestone M1 is in place, since older
archives cannot be read by later code.
"""

import argparse
import json
import os
import time

import numpy as np

import lanfear as lf

# Shared with the orbit driver in this directory: root-only set-up that cannot
# leave the other ranks hanging if it fails.
from run_orbits_mpi import run_on_root

M_TOTAL = 1.0  # total mass, in the 1e10 Msun unit of lanfear.Potential.DEFAULT_G
SCALE_RADIUS = 3.0  # long-axis scale length (kpc)

MODELS = {
    "triaxial": {"b_over_a": 0.79, "c_over_a": 0.5},
    "bar": {"b_over_a": 0.45, "c_over_a": 0.35},
}

# Corotation radii in half-mass radii; inf is the static run.
DEFAULT_COROTATION_RADII = [np.inf, 16.0, 8.0, 4.0, 2.0, 1.5, 1.2]


def hernquist_distribution_function(energy):
    """Isotropic Hernquist (1990) distribution function, up to a constant.

    Parameters
    ----------
    energy : numpy.ndarray
        Dimensionless binding energy ``-E a / (G M)``, in [0, 1].

    Returns
    -------
    f : numpy.ndarray
        Unnormalised phase-space density (zero for unbound energies).
    """
    q = np.sqrt(np.clip(energy, 0.0, 1.0 - 1e-12))
    one_minus_q2 = 1.0 - q**2
    f = (
        3.0 * np.arcsin(q)
        + q
        * np.sqrt(one_minus_q2)
        * (1.0 - 2.0 * q**2)
        * (8.0 * q**4 - 8.0 * q**2 - 3.0)
    ) / one_minus_q2**2.5
    return np.clip(f, 0.0, None)


def sample_hernquist_speeds(r, a, rng, chunk_size=16384):
    """Draw speeds from the isotropic Hernquist distribution function.

    At each radius the speed ``v`` has density ``v**2 f(Psi(r) - v**2 / 2)``
    on ``[0, v_esc(r)]``, sampled by rejection against an envelope taken from
    the maximum on a logarithmic grid.

    Parameters
    ----------
    r : numpy.ndarray
        Radii at which to draw one speed each.
    a : float
        Hernquist scale radius.
    rng : numpy.random.Generator
        Random number generator.
    chunk_size : int, optional
        Number of particles handled at once (bounds the memory use).

    Returns
    -------
    speeds : numpy.ndarray
        Speeds in velocity units of ``lanfear.Potential.DEFAULT_G``.
    """
    psi = 1.0 / (1.0 + r / a)  # dimensionless relative potential
    u_grid = np.geomspace(1e-4, 1.0, 256)  # speed in units of v_esc
    fraction_of_escape = np.empty_like(r)
    for start in range(0, len(r), chunk_size):
        psi_chunk = psi[start : start + chunk_size]
        envelope = 1.2 * np.max(
            u_grid**2
            * hernquist_distribution_function(psi_chunk[:, None] * (1.0 - u_grid**2)),
            axis=1,
        )
        u = np.empty_like(psi_chunk)
        pending = np.arange(len(psi_chunk))
        while pending.size:
            trial = rng.uniform(0.0, 1.0, pending.size)
            density = trial**2 * hernquist_distribution_function(
                psi_chunk[pending] * (1.0 - trial**2)
            )
            accept = rng.uniform(0.0, 1.0, pending.size) * envelope[pending] < density
            u[pending[accept]] = trial[accept]
            pending = pending[~accept]
        fraction_of_escape[start : start + chunk_size] = u
    v_escape = np.sqrt(2.0 * psi * lf.Potential.DEFAULT_G * M_TOTAL / a)
    return fraction_of_escape * v_escape


def equivalent_scale_radius(b_over_a, c_over_a):
    """Scale radius ``(a b c)**(1/3)`` of the spherical-equivalent model.

    Parameters
    ----------
    b_over_a, c_over_a : float
        Axis ratios.

    Returns
    -------
    radius : float
        Spherical-equivalent Hernquist scale radius.
    """
    return SCALE_RADIUS * (b_over_a * c_over_a) ** (1.0 / 3.0)


def corotation_pattern_speed(corotation_radius, b_over_a, c_over_a):
    """Pattern speed putting corotation at ``corotation_radius`` half-mass radii.

    Uses the circular frequency of the spherical-equivalent Hernquist model,
    ``Omega_c(r) = sqrt(G M r) / (r (r + a))``, whose half-mass radius is
    ``(1 + sqrt(2)) a``.

    Parameters
    ----------
    corotation_radius : float
        Corotation radius in half-mass radii (``inf`` for a static figure).
    b_over_a, c_over_a : float
        Axis ratios.

    Returns
    -------
    pattern_speed : float
        ``Omega_c(R_CR)`` in velocity unit / length unit (0 if static).
    """
    if not np.isfinite(corotation_radius):
        return 0.0
    a = equivalent_scale_radius(b_over_a, c_over_a)
    r = corotation_radius * (1.0 + np.sqrt(2.0)) * a
    return np.sqrt(lf.Potential.DEFAULT_G * M_TOTAL * r) / (r * (r + a))


def make_system(n_particles, b_over_a, c_over_a, pattern_speed, seed):
    """Sample a tumbling ellipsoidal Hernquist model (inertial frame).

    Positions are exact for the ellipsoidally stratified profile; velocities
    are isotropic draws from the spherical-equivalent distribution function at
    the spherical-equivalent radius, taken as co-rotating-frame velocities, so
    the inertial velocities are ``v + Omega x x``. For one seed, positions and
    co-rotating velocities are the same at every pattern speed.

    Parameters
    ----------
    n_particles : int
        Number of star particles.
    b_over_a, c_over_a : float
        Axis ratios (long axis along x, short along z).
    pattern_speed : float
        Rotation about z (velocity unit / length unit).
    seed : int
        Random seed.

    Returns
    -------
    particles : lanfear.ParticleSystem
        The (unprepared) system.
    """
    rng = np.random.default_rng(seed)
    semi_axes = SCALE_RADIUS * np.array([1.0, b_over_a, c_over_a])
    su = np.sqrt(rng.uniform(0, 1, n_particles))
    m = su / (1 - su)  # inverts M(<m) / M = m^2 / (1 + m)^2

    def directions():
        mu = rng.uniform(-1, 1, n_particles)
        az = rng.uniform(0, 2 * np.pi, n_particles)
        st = np.sqrt(1 - mu**2)
        return np.stack([st * np.cos(az), st * np.sin(az), mu], axis=1)

    positions = m[:, None] * directions() * semi_axes
    a_eq = equivalent_scale_radius(b_over_a, c_over_a)
    speeds = sample_hernquist_speeds(m * a_eq, a_eq, rng)
    velocities = speeds[:, None] * directions()
    velocities += np.cross([0.0, 0.0, pattern_speed], positions)
    return lf.ParticleSystem(
        pos=positions,
        vel=velocities,
        mass=np.full(n_particles, M_TOTAL / n_particles),
        ids=np.arange(n_particles, dtype=np.int64),
        species=np.full(n_particles, "STAR"),
    )


def prepare_system(particles, pattern_speed):
    """Recentre and align, imposing the known pattern speed.

    The true angular velocity ``(0, 0, pattern_speed)`` is set before
    :meth:`~lanfear.ParticleSystem.align`, which rotates it into the aligned
    frame together with the particles (the sign of each aligned axis is
    arbitrary). The single-snapshot estimate is computed afterwards for
    reference only.

    Parameters
    ----------
    particles : lanfear.ParticleSystem
        The system (modified in place).
    pattern_speed : float
        True rotation about the model's z axis.

    Returns
    -------
    estimate : dict
        :meth:`~lanfear.ParticleSystem.estimate_pattern_speed` result.
    """
    particles.recentre(on="shrinking_sphere")
    particles.pattern_speed = np.array([0.0, 0.0, pattern_speed])
    particles.align()
    particles.estimate_scale_radius()
    return particles.estimate_pattern_speed(check_rigidity=False)


def choose_orbits(particles, n_orbits, seed):
    """Select the particles to integrate (the same subset in every run).

    Parameters
    ----------
    particles : lanfear.ParticleSystem
        The prepared system.
    n_orbits : int
        Number of particles to draw.
    seed : int
        Random seed of the draw.

    Returns
    -------
    subset : lanfear.ParticleSystem
        The selected particles.
    """
    if n_orbits >= particles.n_particles:
        return particles
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.sort(particles.ids), n_orbits, replace=False)
    return particles.select(np.isin(particles.ids, chosen))


def output_stem(out_dir, model, corotation_radius, args):
    """Path (without extension) of one run's output.

    Parameters
    ----------
    out_dir : str
        Output directory.
    model : str
        Model name.
    corotation_radius : float
        Corotation radius in half-mass radii (``inf`` if static).
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    stem : str
        Output path stem.
    """
    label = (
        "static" if not np.isfinite(corotation_radius) else f"rcr{corotation_radius:g}"
    )
    return os.path.join(
        out_dir,
        f"rotation_grid_{model}_{label}_n{args.n_particles}_orbits{args.n_orbits}"
        f"_nmax{args.n_max}_lmax{args.l_max}_seed{args.seed}",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=str, required=True, help="output directory")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument(
        "--corotation-radii",
        type=float,
        nargs="+",
        default=DEFAULT_COROTATION_RADII,
        help="corotation radii in half-mass radii ('inf' for the static run)",
    )
    ap.add_argument("--n-particles", type=int, default=500_000)
    ap.add_argument("--n-orbits", type=int, default=100_000)
    ap.add_argument("--n-max", type=int, default=16)
    ap.add_argument("--l-max", type=int, default=4)
    ap.add_argument("--periods", type=int, default=50)
    ap.add_argument("--samples", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()
    lf.set_verbosity("INFO")

    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except Exception:
        comm, rank = None, 0

    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
    for model in args.models:
        shape = MODELS[model]
        for corotation_radius in args.corotation_radii:
            stem = output_stem(args.out_dir, model, corotation_radius, args)
            done = os.path.exists(f"{stem}.npz") if rank == 0 else None
            if comm is not None:
                done = comm.bcast(done, root=0)
            if done:
                if rank == 0:
                    print(f"[root] {stem}.npz exists; skipping", flush=True)
                continue

            omega = corotation_pattern_speed(corotation_radius, **shape)

            def build():
                """Build the model and pick the orbits (root rank only)."""
                t0 = time.perf_counter()
                particles = make_system(
                    args.n_particles, pattern_speed=omega, seed=args.seed, **shape
                )
                estimate = prepare_system(particles, omega)
                potential = lf.Potential.from_particles(
                    particles, n_max=args.n_max, l_max=args.l_max
                )
                subset = choose_orbits(particles, args.n_orbits, args.seed)
                metadata = {
                    "model": model,
                    **shape,
                    "corotation_radius_half_mass": (
                        None
                        if not np.isfinite(corotation_radius)
                        else corotation_radius
                    ),
                    "pattern_speed_true": particles.pattern_speed.tolist(),
                    "pattern_speed_estimate": estimate["pattern_speed"].tolist(),
                    "pattern_speed_estimate_sigma": estimate[
                        "pattern_speed_uncertainty"
                    ].tolist(),
                    "n_particles": args.n_particles,
                    "n_orbits": args.n_orbits,
                    "n_max": args.n_max,
                    "l_max": args.l_max,
                    "periods": args.periods,
                    "samples": args.samples,
                    "seed": args.seed,
                }
                print(
                    f"[root] {model}, R_CR={corotation_radius:g} r_half: true "
                    f"pattern speed {np.round(particles.pattern_speed, 3)}, "
                    f"estimate {np.round(estimate['pattern_speed'], 3)} "
                    f"(built in {time.perf_counter() - t0:.1f} s)",
                    flush=True,
                )
                return potential, subset, metadata

            built = run_on_root(comm, rank, build)
            potential, subset, metadata = built if rank == 0 else (None, None, None)
            results = lf.analyse_family(
                potential,
                subset,
                family="STAR",
                n_periods=args.periods,
                n_samples=args.samples,
                comm="auto",
            )
            if rank == 0:
                path = results.save(f"{stem}.npz")
                with open(f"{stem}.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                print(f"[root] saved {path}", flush=True)


if __name__ == "__main__":
    main()
