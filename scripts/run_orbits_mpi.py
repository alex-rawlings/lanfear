"""Example / smoke test for MPI-parallel orbit integration.

Builds a synthetic Hernquist snapshot on the root rank, constructs the SCF
potential, and integrates + frequency-analyses all star orbits distributed
across MPI ranks. Prints a summary and a checksum; the checksum is independent of
the number of ranks, so running at several rank counts verifies the
decomposition is correct. The saved output carries the frequency data
(fundamentals + spectral lines), so a reloaded result supports the full
classification (rosette/boxlet/irregular) and the frequency map.

    # serial
    python scripts/run_orbits_mpi.py --n 20000

    # parallel (openmpi is loaded by load_py313)
    mpirun -n 4 python scripts/run_orbits_mpi.py --n 20000
    srun   -n 4 python scripts/run_orbits_mpi.py --n 20000

Set OMP_NUM_THREADS for per-rank threading (hybrid MPI+OpenMP).

Rotating figures: by default the pattern speed is estimated from the snapshot
(``--pattern-speed estimate``; 'none' forces a static potential, a number or
an 'x,y,z' vector imposes one). If it rotates, the potential turns rigidly
during the integration, each orbit near corotation is integrated for longer to
span ``--periods`` body-frame periods (up to ``--max-body-period-factor``
times longer), and a summary of the pattern-speed ratio
``epsilon = Omega_p / Omega_c``, the near-corotation fraction and the runaway
orbits is printed. The output file name records a non-default pattern-speed
choice so a static re-run does not overwrite the rotating one.

    # a rotating-figure snapshot without a black hole (centred automatically)
    srun -n 80 python scripts/run_orbits_mpi.py --file snap_030.hdf5 --r-max 30
"""

import argparse
import os
import tempfile
import time
import numpy as np
import lanfear as lf


def make_dummy_snapshot(path, n, a=3.0, m_total=1e10, seed=5):
    import h5py

    rng = np.random.default_rng(seed)
    su = np.sqrt(rng.uniform(0, 1, n))
    r = a * su / (1 - su)
    mu = rng.uniform(-1, 1, n)
    az = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - mu**2)
    pos = np.stack([r * st * np.cos(az), r * st * np.sin(az), r * mu], axis=1)
    Menc = m_total * (r / (r + a)) ** 2
    vc = np.sqrt(lf.Potential.DEFAULT_G * Menc / np.maximum(r, 1e-6))
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    vel = np.stack([-vc * np.sin(phi), vc * np.cos(phi), np.zeros(n)], axis=1)
    with h5py.File(path, "w") as f:
        f.create_group("Header").attrs["MassTable"] = np.zeros(6)
        g = f.create_group("PartType4")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=vel)
        g.create_dataset("Masses", data=np.full(n, m_total / n))
        g.create_dataset("ParticleIDs", data=np.arange(n, dtype=np.int64))


def pattern_speed_arg(value):
    """Parse --pattern-speed.

    Parameters
    ----------
    value : str
        'estimate', 'none', a number (rotation about the aligned short axis z)
        or a comma-separated 'x,y,z' angular-velocity vector (aligned frame),
        in velocity/length units.

    Returns
    -------
    pattern_speed : str, float or numpy.ndarray
        The value in the form :meth:`lanfear.ParticleSystem.prepare` takes.
    """
    if value.lower() in ("estimate", "none"):
        return value.lower()
    if "," in value:
        vector = np.array([float(v) for v in value.split(",")])
        if vector.shape != (3,):
            raise argparse.ArgumentTypeError(
                f"--pattern-speed vector needs 3 components, got {value!r}"
            )
        return vector
    return float(value)


def pattern_speed_label(pattern_speed):
    """Output-file suffix recording a non-default pattern-speed choice.

    Parameters
    ----------
    pattern_speed : str, float or numpy.ndarray
        Parsed ``--pattern-speed`` value.

    Returns
    -------
    suffix : str
        '' for 'estimate', otherwise e.g. '_ps-none' or '_ps-12.5'.
    """
    if isinstance(pattern_speed, str):
        return "" if pattern_speed == "estimate" else f"_ps-{pattern_speed}"
    values = np.atleast_1d(pattern_speed)
    return "_ps-" + "_".join(f"{v:g}" for v in values)


def resolve_centre(particles, centre):
    """Pick the centring method, resolving 'auto'.

    Parameters
    ----------
    particles : lanfear.ParticleSystem
        The loaded snapshot.
    centre : str
        ``--centre`` value.

    Returns
    -------
    centre : str
        'bh' if 'auto' and the snapshot holds a black hole, 'shrinking_sphere'
        if 'auto' without one, otherwise ``centre`` unchanged.
    """
    if centre != "auto":
        return centre
    return "bh" if np.any(particles.species == "BH") else "shrinking_sphere"


def report_rotation(res, corotation_band):
    """Print the figure-rotation summary of an orbit run.

    Parameters
    ----------
    res : lanfear.OrbitResults
        The orbit results.
    corotation_band : tuple of float
        ``(low, high)`` epsilon range counted as near corotation.
    """
    ok = res.ok & ~res.runaway
    epsilon = res.pattern_speed_ratio[ok]
    low, high = corotation_band
    # Periods each orbit was asked for (more if adaptively extended); the
    # body-frame lengthening is the factor on top of that.
    periods = res.n_periods if res.n_periods_used is None else res.n_periods_used
    periods = np.broadcast_to(periods, ok.shape)[ok]
    cycles = res.body_periods_integrated[ok]
    lengthening = res.column("t_total")[ok] / (periods * res.column("period")[ok])
    print(
        f"[root] figure rotates at {np.round(res.pattern_speed, 4)} "
        f"(|Omega_p| = {np.linalg.norm(res.pattern_speed):.4g})",
        flush=True,
    )
    print(
        f"[root] epsilon = Omega_p / Omega_c percentiles 5/50/95: "
        f"{np.round(np.percentile(epsilon, [5, 50, 95]), 3)}; "
        f"{100 * np.mean((epsilon >= low) & (epsilon <= high)):.1f}% near "
        f"corotation ({low:g}-{high:g}), {100 * np.mean(epsilon > high):.1f}% "
        f"beyond",
        flush=True,
    )
    print(
        f"[root] {100 * np.mean(lengthening > 1.5):.1f}% of orbits integrated "
        f"for longer to span body-frame periods (mean x{np.mean(lengthening):.2f}); "
        f"{100 * np.mean(cycles < periods / np.sqrt(2)):.1f}% still short "
        f"of their body-frame periods; "
        f"{int(res.runaway.sum())} runaway orbits",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, help="Gadget snapshot file")
    ap.add_argument("--n", type=int, default=20000, help="number of star particles")
    ap.add_argument("--periods", type=int, default=50)
    ap.add_argument("--samples", type=int, default=2048)
    ap.add_argument("--n-lines", type=int, default=4, help="spectral lines per axis")
    ap.add_argument("--n-max", type=int, default=16)
    ap.add_argument("--l-max", type=int, default=4)
    ap.add_argument(
        "--r-max",
        type=float,
        default=None,
        help="only integrate particles within this radius (HO/physical units of "
        "the recentred system); the potential is still built from all particles",
    )
    ap.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="integrate a random subset of this many particles (default: all); "
        "the potential is still built from all particles",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for the --subsample draw (kept fixed so the checksum stays "
        "independent of the rank count)",
    )
    ap.add_argument(
        "--fam",
        type=str,
        help="particle family to integrate",
        default="STAR",
        choices=["STAR", "DM"],
    )
    ap.add_argument(
        "--centre",
        type=str,
        help="centre method; 'auto' (default) centres on the black hole(s) if "
        "the snapshot has any, otherwise with the shrinking sphere",
        default="auto",
        choices=["auto", "bh", "shrinking_sphere", "field", "STAR", "DM"],
    )
    ap.add_argument(
        "--pattern-speed",
        type=pattern_speed_arg,
        default="estimate",
        help="figure pattern speed: 'estimate' (default) to measure it from the "
        "snapshot, 'none' for a static potential, a number (rotation about the "
        "aligned short axis) or an 'x,y,z' vector in the aligned frame "
        "(velocity/length units)",
    )
    ap.add_argument(
        "--max-body-period-factor",
        type=int,
        default=lf.orbits.MAX_BODY_PERIOD_FACTOR,
        help="for a rotating figure, integrate orbits near corotation up to this "
        "many times longer (a power of two) so they span --periods body-frame "
        "periods; 1 disables. Multiplies with --max-extensions lengthening",
    )
    ap.add_argument(
        "--runaway-factor",
        type=float,
        default=10.0,
        help="flag orbits whose maximum radius exceeds this many times the larger "
        "of their initial radius and the system radius as runaways",
    )
    ap.add_argument(
        "--max-extensions",
        type=int,
        default=5,
        help="re-integrate orbits whose frequency drift exceeds "
        "--diffusion-threshold for --extension-factor times longer, up to this "
        "many times (0 = off)",
    )
    ap.add_argument(
        "--extension-factor",
        type=float,
        default=2.0,
        help="growth factor of the integration length per extension round",
    )
    ap.add_argument(
        "--diffusion-threshold",
        type=float,
        default=0.1,
        help="diffusion rate above which an orbit is extended",
    )
    ap.add_argument(
        "--diffusion-drop",
        type=float,
        default=0.5,
        help="an extended orbit whose rate is at least this fraction of its "
        "previous rate is judged chaotic",
    )
    args = ap.parse_args()
    lf.set_verbosity("INFO")
    POT_TOL = 0.001  # 0.1% potential target agreement (median)

    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
    except Exception:
        rank, size = 0, 1

    potential = particles = to_integrate = None
    if rank == 0:
        lf.print_package_info()
        d = tempfile.mkdtemp()
        outfile = f"lanfear_orbits/orbits_{args.fam}.npz"
        if args.file is not None:
            path = args.file
            _dname, _ext = os.path.splitext(outfile)
            outfile = os.path.join(
                os.path.dirname(path),
                f"{_dname}_{os.path.basename(path).replace('.hdf5', _ext)}",
            )
        else:
            path = os.path.join(d, "snap.hdf5")
            make_dummy_snapshot(path, args.n)
        _dname, _ext = os.path.splitext(outfile)
        outfile = f"{_dname}{pattern_speed_label(args.pattern_speed)}{_ext}"
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        particles = lf.ParticleSystem.from_gadget_hdf5(path)
        centre = resolve_centre(particles, args.centre)
        print(f"[root] centring with '{centre}'", flush=True)
        particles.prepare(centre=centre, pattern_speed=args.pattern_speed)
        potential = lf.Potential.from_particles(
            particles, n_max=args.n_max, l_max=args.l_max
        )
        pot_result = potential.validate()
        assert pot_result.passed(
            POT_TOL
        ), f"median relerr {pot_result.median:.3%} exceeds tolerance {POT_TOL:.1%}"
        print(f"\nPASS: median agreement {pot_result.median:.3%} < {POT_TOL:.1%}")
        # The potential is built from all particles above; optionally restrict
        # which orbits are integrated to those within --r-max (the potential is
        # unchanged). radius_mask composes with select like species_mask.
        to_integrate = particles
        if args.r_max is not None:
            to_integrate = particles.select(particles.radius_mask(args.r_max))
            print(
                f"[root] restricting integration to r < {args.r_max:g}: "
                f"{to_integrate.n_particles} of {particles.n_particles} particles",
                flush=True,
            )
        if args.subsample is not None:
            before = to_integrate.n_particles
            to_integrate = to_integrate.random_subset(
                args.subsample, rng=np.random.default_rng(args.seed)
            )
            print(
                f"[root] subsampling {to_integrate.n_particles} of {before} "
                f"particles for integration (seed={args.seed})",
                flush=True,
            )
        print(
            f"[root] {particles.n_particles} particles, scale_radius={particles.scale_radius:.3f}, "
            f"threads={os.environ.get('OMP_NUM_THREADS', '?')}",
            flush=True,
        )

    t0 = time.perf_counter()
    res = lf.analyse_family(
        potential,
        to_integrate,
        family=args.fam,
        n_periods=args.periods,
        n_samples=args.samples,
        n_lines=args.n_lines,
        comm="auto",
        max_extensions=args.max_extensions,
        extension_factor=args.extension_factor,
        diffusion_threshold=args.diffusion_threshold,
        diffusion_drop=args.diffusion_drop,
        runaway_factor=args.runaway_factor,
        max_body_period_factor=args.max_body_period_factor,
    )
    dt = time.perf_counter() - t0

    if rank == 0:
        ok = res.ok
        # Checksum over the robust columns (exclude tiny-noise energy values) so
        # it is bit-reproducible across rank counts.
        checksum = float(np.sum(res.column("r_max")) + np.sum(res.column("period")))
        print(
            f"[{size} rank(s)] {len(res.ids)} orbits in {dt:.2f}s | "
            f"{np.mean(ok):.2%} ok | "
            f"median energy_drift={np.median(res.column('energy_drift')[ok]):.2e} | "
            f"checksum={checksum:.10e}",
            flush=True,
        )
        if np.any(res.pattern_speed):
            report_rotation(res, lf.classify.DEFAULT_COROTATION_BAND)

        # save output
        res.save(outfile)
        print(f"[root] saved {outfile}", flush=True)


if __name__ == "__main__":
    main()
