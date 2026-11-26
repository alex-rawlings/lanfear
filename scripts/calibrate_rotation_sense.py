"""Calibrate and validate the rotation-sense labels (lanfear.rotation_sense).

Two modes, both MPI-parallel over the orbits under ``srun``:

``ellipsoid``
    Runs lanfear on a perfect-ellipsoid model from
    ``make_perfect_ellipsoid.py`` and scores the rotation-sense measures
    against its exact reference families:

    * the circulation coherence ``kappa`` of boxes and tubes, and a scan of the
      ``coherence_min`` cut between them (static figure only: in a tumbling
      figure boxes precess with the pattern and carry a net inertial spin, so
      they are not expected to have ``kappa`` near 0);
    * a scan of the ``perpendicular`` band, for a reference along +z, between
      short-axis tubes (along z) and long-axis tubes (perpendicular);
    * the co/counter labels about +z against the truth: in a tumbling figure
      the sign of each orbit's net inertial winding about z, measured by
      AGAMA; in a static one the sign of each short-axis tube's initial
      ``L_z`` (the static model reverses a random half of the velocities, so
      AGAMA's winding of the parent orbit does not give a particle's sense).

``mocks``
    Builds synthetic systems with known kinematics, fits and integrates them,
    and checks the labels against the sign of each orbit's initial ``L . n``:

    * ``counter_fraction``: a rotating sphere with 30% of its particles
      reversed;
    * ``misaligned``: a stellar disc spinning about z in a dark halo spinning
      about an axis 40 degrees away, labelled against the "STAR" and "DM"
      references;
    * ``counter_core``: a sphere whose inner 2 length units counter-rotate,
      labelled against "STAR_r<2" and "STAR".

Integrations are saved in ``--out`` and reused unless ``--reintegrate``. Each
mode writes a JSON summary next to them.

    srun python calibrate_rotation_sense.py ellipsoid \
        --dir /orion/ptmp/arawling/orbits/agama --out OUT
    srun python calibrate_rotation_sense.py mocks --out OUT
"""

import argparse
import json
import os

import numpy as np

import lanfear as lf
from lanfear.rotation_sense import RotationSense

REFERENCE_FAMILIES = (
    "box",
    "short_axis_tube",
    "inner_long_axis_tube",
    "outer_long_axis_tube",
)
COHERENCE_GRID = np.round(np.arange(0.02, 0.92, 0.02), 2)
PERPENDICULAR_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
G = 43009.1  # kpc (km/s)^2 / 1e10 Msun


def _rank() -> int:
    try:
        from mpi4py import MPI

        return MPI.COMM_WORLD.Get_rank()
    except Exception:
        return 0


def _percentiles(values) -> dict:
    values = np.asarray(values)[np.isfinite(values)]
    if not len(values):
        return {}
    q = np.percentile(values, [5, 25, 50, 75, 95])
    return {f"p{p}": float(v) for p, v in zip((5, 25, 50, 75, 95), q)}


def _balanced_scan(low, high, grid):
    """Balanced accuracy of ``value < t`` for ``low`` and ``>= t`` for ``high``."""
    if not (len(low) and len(high)):
        return [], None
    rows = []
    for t in grid:
        accuracy = 0.5 * (np.mean(low < t) + np.mean(high >= t))
        rows.append((float(t), float(accuracy)))
    best = max(rows, key=lambda row: row[1])
    return rows, best


def _integrate(
    potential, particles, out_path, family, n_periods, n_samples, reintegrate
):
    """Integrate (all ranks) or load (root) one population."""
    rank = _rank()
    if os.path.exists(out_path) and not reintegrate:
        return lf.OrbitResults.load(out_path) if rank == 0 else None
    results = lf.analyse_family(
        potential,
        particles,
        family=family,
        n_periods=n_periods,
        n_samples=n_samples,
        comm="auto",
    )
    if rank == 0:
        results.save(out_path)
    return results


# ------------------------------------------------------------------ ellipsoid
def run_ellipsoid(args) -> None:
    rank = _rank()
    with open(os.path.join(args.dir, "perfect_ellipsoid_model.json")) as f:
        model = json.load(f)
    label = os.path.basename(os.path.normpath(args.dir))
    out_path = os.path.join(args.out, f"orbits_{label}_{args.n_orbits}.npz")
    potential = subset = ps = None
    cached = os.path.exists(out_path) and not args.reintegrate
    if rank == 0:
        ps = lf.ParticleSystem.from_gadget_hdf5(
            os.path.join(args.dir, "perfect_ellipsoid_ics.hdf5")
        )
        has_bh = bool(np.any(ps.species == "BH"))
        ps.prepare(
            centre="bh_all" if has_bh else "shrinking_sphere",
            pattern_speed=model["pattern_speed"] or "none",
        )
        if not cached:
            potential = lf.Potential.from_particles(
                ps, n_max=args.n_max, l_max=args.l_max
            )
            subset = ps.random_subset(
                args.n_orbits, rng=np.random.default_rng(args.seed)
            )
    results = _integrate(
        potential,
        subset,
        out_path,
        "STAR",
        args.periods,
        args.samples,
        args.reintegrate,
    )
    if rank != 0:
        return

    results = results.drop_runaways()
    if "z" not in results.references:
        results.add_reference(lf.SpinReference.from_axis([0, 0, 1], results.frame, "z"))
    field = ps.reference_spin()
    reference = np.load(os.path.join(args.dir, "perfect_ellipsoid_reference.npz"))
    row_of_id = {int(i): k for k, i in enumerate(reference["particle_ids"].tolist())}
    rows = np.array([row_of_id[int(i)] for i in results.ids])
    family = reference["particle_family"][rows].astype(str)
    regular = reference["particle_lyapunov"][rows] <= 0
    turns = reference["particle_inertial_turns_z"][rows]
    tumbling = bool(model["pattern_speed"])

    ok = results.ok & regular & np.isin(family, REFERENCE_FAMILIES)
    kappa = results.circulation_coherence()
    cosine = results.reference_cosine("z")
    summary = {
        "model": label,
        "tumbling": tumbling,
        "pattern_speed": model["pattern_speed"],
        "n_orbits": int(len(results.ids)),
        "n_scored": int(ok.sum()),
        "field_spin": {
            "axis": field.axis.tolist(),
            "angle_uncertainty_deg": float(np.degrees(field.angle_uncertainty)),
            "significant": field.significant,
        },
        "coherence": {
            f: _percentiles(kappa[ok & (family == f)]) for f in REFERENCE_FAMILIES
        },
        "abs_cosine_z": {
            f: _percentiles(np.abs(cosine[ok & (family == f)]))
            for f in REFERENCE_FAMILIES
        },
    }

    boxes = ok & (family == "box")
    tubes = ok & (family != "box")
    scan, best = _balanced_scan(kappa[boxes], kappa[tubes], COHERENCE_GRID)
    summary["coherence_scan"] = {"grid": scan, "best": best}

    short = ok & (family == "short_axis_tube")
    long_axis = ok & np.isin(family, ("inner_long_axis_tube", "outer_long_axis_tube"))
    scan, best = _balanced_scan(
        np.abs(cosine[long_axis]), np.abs(cosine[short]), PERPENDICULAR_GRID
    )
    summary["perpendicular_scan"] = {"grid": scan, "best": best}

    # Co/counter about +z against the truth (see the module notes), for orbits
    # that lanfear calls co/counter.
    sense = results.rotation_sense("z")
    if tumbling:
        truth_name = "AGAMA net inertial winding about z (|turns| >= 1)"
        scored = ok & (np.abs(turns) >= 1)
        truth = np.where(turns > 0, RotationSense.CO, RotationSense.COUNTER)
    else:
        truth_name = "initial L_z sign of reference short-axis tubes"
        order = np.argsort(ps.ids)
        index = order[np.searchsorted(ps.ids[order], results.ids)]
        lz0 = np.cross(ps.pos[index], ps.vel[index])[:, 2]
        scored = ok & (family == "short_axis_tube")
        truth = np.where(lz0 > 0, RotationSense.CO, RotationSense.COUNTER)
    labelled = scored & np.isin(sense, (RotationSense.CO, RotationSense.COUNTER))
    summary["sense_vs_truth"] = {
        "truth": truth_name,
        "n_scored": int(scored.sum()),
        "n_labelled": int(labelled.sum()),
        "agreement": float(np.mean(sense[labelled] == truth[labelled])),
        "by_family": {
            f: {
                "n": int(np.sum(labelled & (family == f))),
                "agreement": (
                    float(
                        np.mean(
                            sense[labelled & (family == f)]
                            == truth[labelled & (family == f)]
                        )
                    )
                    if np.any(labelled & (family == f))
                    else None
                ),
            }
            for f in REFERENCE_FAMILIES
        },
        "label_counts": {
            lf.ROTATION_SENSE_NAMES[int(v)]: int(np.sum(sense[ok] == v))
            for v in np.unique(sense[ok])
        },
    }
    _report(summary, os.path.join(args.out, f"summary_{label}.json"))


# ---------------------------------------------------------------------- mocks
def _hernquist(n, rng, scale=3.0):
    """Hernquist positions (truncated at 30 scale radii)."""
    su = np.sqrt(rng.uniform(0, 0.97, n))
    r = scale * su / (1.0 - su)
    u = rng.normal(size=(n, 3))
    return r[:, None] * u / np.linalg.norm(u, axis=1)[:, None]


def _circular_speed(pos, mass):
    """Circular speed of the spherically averaged enclosed mass, per particle."""
    r = np.linalg.norm(pos, axis=1)
    order = np.argsort(r)
    enclosed = np.empty(len(r))
    enclosed[order] = np.cumsum(mass[order])
    return np.sqrt(G * enclosed / np.maximum(r, 1e-3))


def _rotating_velocities(pos, mass, axis, rng, fraction=0.7, dispersion=0.3):
    """Streaming at ``fraction`` of v_c about ``axis`` plus isotropic noise."""
    vc = _circular_speed(pos, mass)
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    phi_hat = np.cross(axis, pos)
    norm = np.linalg.norm(phi_hat, axis=1)[:, None]
    phi_hat = np.divide(phi_hat, norm, out=np.zeros_like(phi_hat), where=norm > 0)
    return fraction * vc[:, None] * phi_hat + dispersion * vc[:, None] * rng.normal(
        size=pos.shape
    )


def _system(pos, vel, species):
    n = len(pos)
    return lf.ParticleSystem(
        pos=pos, vel=vel, mass=np.full(n, 1.0 / n), ids=np.arange(n), species=species
    )


def _mock(name, rng):
    """Particles and the references to test for one mock."""
    tilt = np.radians(40.0)
    if name == "counter_fraction":
        pos = _hernquist(200_000, rng)
        vel = _rotating_velocities(pos, np.full(len(pos), 1 / len(pos)), [0, 0, 1], rng)
        vel[rng.uniform(size=len(pos)) < 0.3] *= -1.0
        return _system(pos, vel, np.full(len(pos), "STAR")), [{"family": "STAR"}]
    if name == "misaligned":
        n_star, n_dm = 60_000, 200_000
        R = rng.gamma(2.0, 2.0, n_star)
        phi = rng.uniform(0, 2 * np.pi, n_star)
        z = 0.3 * np.arctanh(rng.uniform(-1, 1, n_star))
        star = np.stack([R * np.cos(phi), R * np.sin(phi), z], axis=1)
        dm = _hernquist(n_dm, rng, scale=10.0)
        pos = np.concatenate([star, dm])
        mass = np.full(len(pos), 1 / len(pos))
        dm_axis = [0.0, -np.sin(tilt), np.cos(tilt)]
        vel = np.concatenate(
            [
                _rotating_velocities(star, mass[:n_star], [0, 0, 1], rng, 0.9, 0.1),
                _rotating_velocities(dm, mass[n_star:], dm_axis, rng, 0.5, 0.3),
            ]
        )
        species = np.concatenate([np.full(n_star, "STAR"), np.full(n_dm, "DM")])
        return _system(pos, vel, species), [{"family": "STAR"}, {"family": "DM"}]
    if name == "counter_core":
        pos = _hernquist(200_000, rng)
        vel = _rotating_velocities(pos, np.full(len(pos), 1 / len(pos)), [0, 0, 1], rng)
        vel[np.linalg.norm(pos, axis=1) < 2.0] *= -1.0
        return _system(pos, vel, np.full(len(pos), "STAR")), [
            {"family": "STAR", "radius": 2.0, "geometry": "spherical"},
            {"family": "STAR"},
        ]
    raise ValueError(name)


def _score(results, ps, ref, cos_min=0.6):
    """Agreement of the labels with the sign of each orbit's initial L . n."""
    rows = np.searchsorted(ps.ids, results.ids)
    l0 = np.cross(ps.pos[rows], ps.vel[rows])
    cos0 = l0 @ ref.axis / np.linalg.norm(l0, axis=1)
    sense = results.rotation_sense(ref)
    clear = np.abs(cos0) >= cos_min
    labelled = clear & np.isin(sense, (RotationSense.CO, RotationSense.COUNTER))
    truth = np.where(cos0 > 0, RotationSense.CO, RotationSense.COUNTER)
    counter_true = np.mean(cos0[clear] < 0)
    counter_found = np.mean(sense[labelled] == RotationSense.COUNTER)
    return {
        "axis": ref.axis.tolist(),
        "n_clear": int(clear.sum()),
        "n_labelled": int(labelled.sum()),
        "agreement": float(np.mean(sense[labelled] == truth[labelled])),
        "counter_fraction_initial": float(counter_true),
        "counter_fraction_labelled": float(counter_found),
        "label_counts": {
            lf.ROTATION_SENSE_NAMES[int(v)]: int(np.sum(sense == v))
            for v in np.unique(sense)
        },
    }


def run_mocks(args) -> None:
    rank = _rank()
    summary = {}
    for name in ("counter_fraction", "misaligned", "counter_core"):
        out_path = os.path.join(args.out, f"orbits_mock_{name}_{args.n_orbits}.npz")
        potential = subset = ps = specs = None
        if rank == 0:
            ps, specs = _mock(name, np.random.default_rng(args.seed))
            ps.prepare(
                centre="field", pattern_speed="none", check_figure_rotation=False
            )
            potential = lf.Potential.from_particles(
                ps, n_max=args.n_max, l_max=args.l_max
            )
            subset = ps.random_subset(
                args.n_orbits, rng=np.random.default_rng(args.seed)
            )
        families = ("STAR", "DM") if name == "misaligned" else "STAR"
        results = _integrate(
            potential,
            subset,
            out_path,
            families,
            args.periods,
            args.samples,
            args.reintegrate,
        )
        if rank != 0:
            continue
        results = results.drop_runaways()
        entry = {}
        for spec in specs:
            ref = ps.reference_spin(**spec)
            entry[ref.name] = _score(results, ps, ref)
            if name == "counter_core":
                inner = results.column("r_mean") * results.length_unit < 1.5
                sense = results.rotation_sense(ref)
                entry[ref.name]["inner_orbit_counts"] = {
                    lf.ROTATION_SENSE_NAMES[int(v)]: int(np.sum(sense[inner] == v))
                    for v in np.unique(sense[inner])
                }
        summary[name] = entry
    if rank == 0:
        _report(summary, os.path.join(args.out, "summary_mocks.json"))


def _report(summary, path) -> None:
    text = json.dumps(summary, indent=2)
    print(text)
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    for mode in ("ellipsoid", "mocks"):
        p = sub.add_parser(mode)
        p.add_argument("--out", required=True, help="output directory")
        p.add_argument(
            "--n-orbits", type=int, default=20_000 if mode == "ellipsoid" else 6_000
        )
        p.add_argument("--n-max", type=int, default=24 if mode == "ellipsoid" else 16)
        p.add_argument("--l-max", type=int, default=8 if mode == "ellipsoid" else 6)
        p.add_argument("--periods", type=int, default=50 if mode == "ellipsoid" else 30)
        p.add_argument(
            "--samples", type=int, default=2048 if mode == "ellipsoid" else 1024
        )
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--reintegrate", action="store_true")
        if mode == "ellipsoid":
            p.add_argument(
                "--dir", required=True, help="make_perfect_ellipsoid.py directory"
            )
    args = ap.parse_args()
    lf.set_verbosity("INFO")
    if _rank() == 0:
        os.makedirs(args.out, exist_ok=True)
    if args.mode == "ellipsoid":
        run_ellipsoid(args)
    else:
        run_mocks(args)


if __name__ == "__main__":
    main()
