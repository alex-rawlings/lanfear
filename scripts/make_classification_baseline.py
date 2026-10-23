"""Freeze, or compare against, a static reference classification.

The classification redesign (``AI_output/classification_milestones.md``) must
leave static-potential labels essentially unchanged. This script builds a fixed
static triaxial population (an SCF fit to a squashed Hernquist sample, as in
``tests/test_classify.py``), integrates and classifies it, and saves the labels
and diagnostics as a baseline, together with the labels of the pinned orbits of
``test_known_orbits``. Everything is generated from seeds stored in the
baseline, so ``--compare`` rebuilds the identical population with the *current*
code and reports how many labels changed, and to what. It does not depend on
the summary-column layout of the code that wrote the baseline.

    # write test_scripts/data/classification_baseline/baseline.npz
    python scripts/make_classification_baseline.py --out test_scripts/data/classification_baseline

    # re-run with the current code and report label agreement
    python scripts/make_classification_baseline.py \
        --compare test_scripts/data/classification_baseline/baseline.npz

The integration is OpenMP-parallel (set OMP_NUM_THREADS); 20000 orbits take a
few seconds to minutes depending on the core count.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

import lanfear as lf
from lanfear import _core
from lanfear.orbits import SUMMARY_COLUMNS, OrbitResults

# Population defaults (stored in the baseline, so --compare reuses them).
DEFAULT_CONFIG = {
    "axis_ratios": [1.0, 0.8, 0.6],
    "n_particles": 300_000,
    "potential_seed": 3,
    "n_max": 12,
    "l_max": 6,
    "n_orbits": 20_000,
    "orbit_seed": 11,
    "n_periods": 50,
    "n_samples": 2048,
    "n_lines": 4,
}

# Pinned initial conditions of tests/test_classify.py::test_known_orbits, with
# the potential each lives in and the label it must keep.
TRIAXIAL_KNOWN = {"axis_ratios": [1.0, 0.75, 0.5], "l_max": 6}
SPHERICAL_KNOWN = {"axis_ratios": [1.0, 1.0, 1.0], "l_max": 4}
KNOWN_ORBITS = {
    "box": (TRIAXIAL_KNOWN, [1.5, 1.0, 0.7, 0.0, 0.0, 0.0]),
    "short_axis_tube": (TRIAXIAL_KNOWN, [2.0, 0.0, 0.6, 0.0, 0.6, 0.15]),
    "inner_long_axis_tube": (
        TRIAXIAL_KNOWN,
        [-3.011, 0.766, 3.396, -0.192, -0.035, 0.203],
    ),
    "outer_long_axis_tube": (
        TRIAXIAL_KNOWN,
        [1.727, -0.851, 3.367, -0.257, -0.015, -0.286],
    ),
    # Velocity in units of the local circular speed (scaled in known_states).
    "rosette": (SPHERICAL_KNOWN, [2.0, 0.0, 0.3, 0.0, 0.9, 0.2]),
}
KNOWN_N_PERIODS = 40
KNOWN_N_SAMPLES = 4096


def build_potential(axis_ratios, n_particles, seed, n_max, l_max):
    """Fit an HO SCF potential to a squashed Hernquist sample (HO units).

    Parameters
    ----------
    axis_ratios : sequence of float
        (3,) factors applied to x, y, z of a spherical Hernquist sample.
    n_particles : int
        Sample size.
    seed : int
        Random seed of the sample.
    n_max, l_max : int
        Expansion orders.

    Returns
    -------
    scf : lanfear._core.SCFPotential
        The fitted potential.
    """
    rng = np.random.default_rng(seed)
    su = np.sqrt(rng.uniform(0, 1, n_particles))
    r = su / (1 - su)
    mu = rng.uniform(-1, 1, n_particles)
    az = rng.uniform(0, 2 * np.pi, n_particles)
    st = np.sqrt(1 - mu**2)
    pos = np.stack([r * st * np.cos(az), r * st * np.sin(az), r * mu], axis=1)
    pos *= np.asarray(axis_ratios)
    return _core.SCFPotential(
        n_max, l_max, pos, np.full(n_particles, 1.0 / n_particles)
    )


def circular_speed(scf, positions):
    """Local circular speed ``sqrt(r a_r)`` at each position (HO units).

    Parameters
    ----------
    scf : lanfear._core.SCFPotential
        The potential.
    positions : numpy.ndarray
        (N, 3) positions.

    Returns
    -------
    speed : numpy.ndarray
        (N,) circular speeds (floored at a small positive value).
    """
    acc = np.asarray(scf.acceleration_batch(np.ascontiguousarray(positions)))
    r = np.linalg.norm(positions, axis=1)
    a_radial = -np.sum(acc * positions, axis=1) / r
    return np.sqrt(np.maximum(a_radial, 1e-6) * r)


def population_states(scf, n_orbits, seed):
    """Random bound-ish initial conditions spanning the box and tube regions.

    Positions uniform in a cube, kept for ``0.5 < r < 5``; velocities in a
    random direction with a speed uniform in ``[0, v_c]`` (the scheme of
    ``tests/test_classify.py::test_population``, vectorised).

    Parameters
    ----------
    scf : lanfear._core.SCFPotential
        The potential.
    n_orbits : int
        Number of initial conditions.
    seed : int
        Random seed.

    Returns
    -------
    states : numpy.ndarray
        (n_orbits, 6) initial states.
    """
    rng = np.random.default_rng(seed)
    positions = np.empty((0, 3))
    while len(positions) < n_orbits:
        trial = rng.uniform(-3, 3, (2 * n_orbits, 3))
        r = np.linalg.norm(trial, axis=1)
        positions = np.concatenate([positions, trial[(r > 0.5) & (r < 5)]])
    positions = positions[:n_orbits]
    direction = rng.uniform(-1, 1, (n_orbits, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-9
    speed = rng.uniform(0, 1.0, n_orbits) * circular_speed(scf, positions)
    return np.concatenate([positions, direction * speed[:, None]], axis=1)


def analyse(scf, states, n_periods, n_samples, n_lines):
    """Integrate, frequency-analyse and wrap a batch of states.

    Parameters
    ----------
    scf : lanfear._core.SCFPotential
        The (static) potential.
    states : numpy.ndarray
        (N, 6) initial states (HO units).
    n_periods, n_samples, n_lines : int
        Integration and analysis settings.

    Returns
    -------
    results : lanfear.OrbitResults
        The orbit results (HO units).
    """
    summary, fundamentals, lines, diffusion = scf.analyse_batch(
        np.ascontiguousarray(states),
        n_periods=n_periods,
        n_samples=n_samples,
        n_lines=n_lines,
    )
    return OrbitResults(
        ids=np.arange(len(states)),
        summary=summary,
        columns=SUMMARY_COLUMNS,
        time_unit=1.0,
        length_unit=1.0,
        n_periods=n_periods,
        n_samples=n_samples,
        initial_radius=np.linalg.norm(states[:, :3], axis=1),
        fundamentals=fundamentals,
        lines=lines,
        diffusion=diffusion,
        pattern_speed=np.zeros(3),
    )


def classify_known_orbits(config):
    """Classify the pinned orbits of ``test_known_orbits``.

    Parameters
    ----------
    config : dict
        Population configuration (supplies the particle count, seed and
        ``n_max`` of the potentials).

    Returns
    -------
    names : dict
        Orbit name -> assigned class name.
    """
    potentials = {}
    names = {}
    for name, (setup, state) in KNOWN_ORBITS.items():
        key = (tuple(setup["axis_ratios"]), setup["l_max"])
        if key not in potentials:
            potentials[key] = build_potential(
                setup["axis_ratios"],
                config["n_particles"],
                config["potential_seed"],
                config["n_max"],
                setup["l_max"],
            )
        scf = potentials[key]
        state = np.array(state, dtype=float)
        if name == "rosette":
            state[3:] *= circular_speed(scf, state[None, :3])[0]
        res = analyse(scf, state[None, :], KNOWN_N_PERIODS, KNOWN_N_SAMPLES, 4)
        names[name] = str(res.classify(drop_runaways=False).names[0])
    return names


def run(config):
    """Build, integrate and classify the baseline population.

    Parameters
    ----------
    config : dict
        Population configuration (see :data:`DEFAULT_CONFIG`).

    Returns
    -------
    results : lanfear.OrbitResults
        The orbit results.
    classification : lanfear.OrbitClassification
        Their classification (runaways kept, so it aligns with ``results``).
    known : dict
        Pinned-orbit name -> class name.
    """
    t0 = time.perf_counter()
    scf = build_potential(
        config["axis_ratios"],
        config["n_particles"],
        config["potential_seed"],
        config["n_max"],
        config["l_max"],
    )
    states = population_states(scf, config["n_orbits"], config["orbit_seed"])
    results = analyse(
        scf, states, config["n_periods"], config["n_samples"], config["n_lines"]
    )
    classification = results.classify(drop_runaways=False)
    known = classify_known_orbits(config)
    print(
        f"{len(states)} orbits integrated and classified in "
        f"{time.perf_counter() - t0:.1f} s",
        flush=True,
    )
    return results, classification, known


def git_commit():
    """Current git commit of the lanfear checkout (or 'unknown').

    Returns
    -------
    commit : str
        The commit hash, with ``-dirty`` appended for uncommitted changes.
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        commit = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def save_baseline(out_dir, config):
    """Run the baseline and write ``baseline.npz`` and ``baseline_orbits.npz``.

    Parameters
    ----------
    out_dir : str
        Output directory (created if needed).
    config : dict
        Population configuration.
    """
    os.makedirs(out_dir, exist_ok=True)
    results, classification, known = run(config)
    path = os.path.join(out_dir, "baseline.npz")
    np.savez(
        path,
        labels=classification.labels,
        names=classification.names.astype(str),
        ids=classification.ids,
        circulation=classification.circulation,
        tube_axis=classification.tube_axis,
        planarity=classification.planarity,
        resonance=classification.resonance,
        resonance_order=classification.resonance_order,
        known_orbits=np.array(list(known)),
        known_names=np.array(list(known.values())),
        config=np.array(json.dumps(config)),
        git_commit=np.array(git_commit()),
    )
    orbits_path = results.save(os.path.join(out_dir, "baseline_orbits.npz"))
    print(f"class counts: {classification.counts()}")
    print(f"pinned orbits: {known}")
    print(f"saved {path} and {orbits_path}")


def compare_baseline(path):
    """Re-run the baseline population with the current code and compare labels.

    Parameters
    ----------
    path : str
        A ``baseline.npz`` written by :func:`save_baseline`.

    Returns
    -------
    status : int
        0 if every pinned orbit kept its label, else 1 (population changes are
        reported, not judged: the acceptable agreement is a milestone decision).
    """
    with np.load(path, allow_pickle=False) as f:
        config = json.loads(str(f["config"]))
        names_before = f["names"].astype(str)
        known_before = dict(
            zip(f["known_orbits"].astype(str), f["known_names"].astype(str))
        )
        commit = str(f["git_commit"])
    print(f"baseline written at commit {commit}; current {git_commit()}")
    _, classification, known = run(config)
    names_after = classification.names.astype(str)
    changed = names_before != names_after
    print(
        f"label agreement: {100 * (1 - changed.mean()):.3f}% "
        f"({int(changed.sum())} of {len(changed)} orbits changed)"
    )
    if changed.any():
        pairs, counts = np.unique(
            np.stack([names_before[changed], names_after[changed]], axis=1),
            axis=0,
            return_counts=True,
        )
        for (before, after), count in sorted(
            zip(map(tuple, pairs), counts), key=lambda item: -item[1]
        ):
            print(f"  {before:>22s} -> {after:<22s} {count}")
    status = 0
    for name, before in known_before.items():
        after = known.get(name)
        flag = "ok" if after == before else "CHANGED"
        if after != before:
            status = 1
        print(f"  pinned {name:<22s} {before} -> {after} [{flag}]")
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=str, help="directory to write the baseline to")
    group.add_argument("--compare", type=str, help="baseline.npz to compare against")
    ap.add_argument("--n-orbits", type=int, default=DEFAULT_CONFIG["n_orbits"])
    ap.add_argument("--n-periods", type=int, default=DEFAULT_CONFIG["n_periods"])
    ap.add_argument("--n-samples", type=int, default=DEFAULT_CONFIG["n_samples"])
    args = ap.parse_args()
    lf.set_verbosity("WARNING")
    if args.compare:
        sys.exit(compare_baseline(args.compare))
    config = dict(
        DEFAULT_CONFIG,
        n_orbits=args.n_orbits,
        n_periods=args.n_periods,
        n_samples=args.n_samples,
    )
    save_baseline(args.out, config)


if __name__ == "__main__":
    main()
