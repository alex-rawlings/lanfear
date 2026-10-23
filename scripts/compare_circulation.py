"""Compare circulation diagnostics across a grid of rotating figures.

Diagnostic for the classification redesign (``AI_output/classification_
milestones.md``, M2): how do the angular-momentum circulation ``circ_a`` (used
by the current classifier), the winding circulation ``|phi_net| / phi_range``,
the azimuthal-to-radial frequency ratio and the hole size behave as a figure
tumbles faster?

It reads the runs written by ``scripts/make_rotation_grid.py``. Every run of a
model integrates the same particles (same positions and co-rotating-frame
velocities), so each orbit keeps the family label it gets in the model's
*static* run as a reference, and the diagnostics of the same orbits can be
followed as the corotation radius shrinks. The static label is a reference,
not the truth: the rotating orbit is a different orbit, and some families
(long-axis tubes in particular) genuinely do not survive the rotation.

Per run it prints, for each static family, the median of each diagnostic about
the family's reference axis (the circulation axis of tubes, z otherwise) and the
fraction of orbits a few candidate tube criteria would call tubes. It also
checks the pattern-locking tolerance: the distribution of
``log10(|Omega_phi,inertial| / |Omega_p|)`` should be bimodal with a gap
around the default ``tol = 1e-3``. With ``--figdir`` it saves:

* ``<model>_diagnostics_vs_rcr.png``: medians against corotation radius;
* ``<model>_circ_vs_winding.png``: ``circ_a`` against winding circulation for
  the static run and the fastest rotation;
* ``<model>_locking.png``: the locking histogram of every rotating run.

    python scripts/compare_circulation.py --grid-dir test_scripts/data/rotation_grid \\
        --figdir figures/compare_circulation
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt

import lanfear as lf
from lanfear.classify import CLASS_NAMES, OrbitClass, OrbitShape

# Candidate thresholds for the printed tube fractions.
CIRC_THRESHOLD = 0.9
WINDING_THRESHOLD = 0.9
SPEED_THRESHOLD = 0.3
HOLE_THRESHOLD = 0.1

FAMILIES = [
    OrbitClass.PIBOX,
    OrbitClass.BOXLET,
    OrbitClass.SHORT_AXIS_TUBE,
    OrbitClass.INNER_LONG_AXIS_TUBE,
    OrbitClass.OUTER_LONG_AXIS_TUBE,
    OrbitClass.ROSETTE,
    OrbitClass.IRREGULAR,
]
LONG_AXIS = (OrbitClass.INNER_LONG_AXIS_TUBE, OrbitClass.OUTER_LONG_AXIS_TUBE)


def find_runs(grid_dir):
    """Group the grid's archives by model, ordered from static to fastest.

    Parameters
    ----------
    grid_dir : str
        Directory written by ``make_rotation_grid.py``.

    Returns
    -------
    runs : dict
        Model name -> list of ``(corotation_radius, npz_path, metadata)``,
        static (``inf``) first, then decreasing corotation radius.
    """
    runs = {}
    for path in glob.glob(os.path.join(grid_dir, "rotation_grid_*.npz")):
        with open(path[: -len(".npz")] + ".json") as f:
            metadata = json.load(f)
        rcr = metadata["corotation_radius_half_mass"]
        rcr = np.inf if rcr is None else float(rcr)
        runs.setdefault(metadata["model"], []).append((rcr, path, metadata))
    for model in runs:
        runs[model].sort(key=lambda item: -item[0])
    return runs


def diagnostics(res):
    """Per-orbit circulation diagnostics of one run.

    Parameters
    ----------
    res : lanfear.OrbitResults
        A run (runaways already dropped).

    Returns
    -------
    values : dict
        ``"circ"``, ``"winding"``, ``"speed"`` (``|Omega_phi| /
        radial_frequency``) and ``"hole"``, each (N, 3) per axis; and
        ``"locking"``, (N,) ``|Omega_phi,inertial| / |Omega_p|`` about the
        pattern axis (NaN if static).
    """
    l_mean = np.stack([res.column(f"L{a}_mean") for a in "xyz"], axis=1)
    l_abs = np.stack([res.column(f"L{a}_abs_mean") for a in "xyz"], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        circ = np.abs(l_mean) / l_abs
        speed = (
            np.abs(res.azimuthal_frequency())
            / np.asarray(res.column("radial_frequency"), dtype=np.float64)[:, None]
        )
    locking = np.full(len(res.ids), np.nan)
    axis = res.pattern_axis()
    if axis is not None:
        a, omega_a = axis
        locking = np.abs(res.azimuthal_frequency("inertial")[:, a]) / abs(omega_a)
    return {
        "circ": circ,
        "winding": res.winding_circulation(),
        "speed": speed,
        "hole": res.hole_fraction(),
        "locking": locking,
    }


def reference_axis(labels):
    """Axis each static family is judged about.

    Parameters
    ----------
    labels : numpy.ndarray
        (N,) static :class:`~lanfear.OrbitClass` labels.

    Returns
    -------
    axis : numpy.ndarray
        (N,) 0 (x) for long-axis tubes, 2 (z) for everything else.
    """
    return np.where(np.isin(labels, LONG_AXIS), 0, 2)


def along(values, axis):
    """Pick one column per row.

    Parameters
    ----------
    values : numpy.ndarray
        (N, 3) per-axis values.
    axis : numpy.ndarray
        (N,) column index per row.

    Returns
    -------
    picked : numpy.ndarray
        (N,) values.
    """
    return np.take_along_axis(values, axis[:, None], axis=1)[:, 0]


def summarise(model, runs):
    """Print the per-family table of every run and collect medians for plots.

    Parameters
    ----------
    model : str
        Model name.
    runs : list
        ``(corotation_radius, path, metadata)`` entries, static first.

    Returns
    -------
    collected : dict
        ``"rcr"`` list, ``"medians"`` {quantity: {family: list}},
        ``"scatter"`` {rcr: (circ, winding, labels)} for the static and
        fastest runs, and ``"locking"`` {rcr: values} for rotating runs.
    """
    static_rcr, static_path, _ = runs[0]
    if np.isfinite(static_rcr):
        raise ValueError(f"model {model!r} has no static run to take labels from")
    static = lf.OrbitResults.load(static_path).drop_runaways()
    static_labels = dict(zip(static.ids, static.classify(drop_runaways=False).labels))

    collected = {
        "rcr": [],
        "medians": {
            q: {f: [] for f in FAMILIES} for q in ("circ", "winding", "speed", "hole")
        },
        "scatter": {},
        "locking": {},
    }
    for rcr, path, metadata in runs:
        res = lf.OrbitResults.load(path).drop_runaways()
        keep = np.isin(res.ids, list(static_labels))
        res = res.select(keep & res.ok)
        labels = np.array([static_labels[i] for i in res.ids])
        values = diagnostics(res)
        axis = reference_axis(labels)
        circ = along(values["circ"], axis)
        winding = along(values["winding"], axis)
        speed = along(values["speed"], axis)
        hole = along(values["hole"], axis)

        label = "static" if not np.isfinite(rcr) else f"R_CR={rcr:g}"
        omega = np.linalg.norm(metadata["pattern_speed_true"])
        print(
            f"\n== {model} {label} (|Omega_p| = {omega:.3g}; {len(res.ids)} orbits) =="
        )
        print(
            f"{'static family':>22s} {'n':>6s} {'circ':>6s} {'wind':>6s} {'speed':>6s} "
            f"{'hole':>6s} | tube by: {'circ':>5s} {'wind':>5s} {'w+spd':>5s} "
            f"{'w+hole':>6s}"
        )
        collected["rcr"].append(rcr)
        for family in FAMILIES:
            sel = labels == family
            n = int(sel.sum())
            med = {
                "circ": np.nanmedian(circ[sel]) if n else np.nan,
                "winding": np.nanmedian(winding[sel]) if n else np.nan,
                "speed": np.nanmedian(speed[sel]) if n else np.nan,
                "hole": np.nanmedian(hole[sel]) if n else np.nan,
            }
            for q, v in med.items():
                collected["medians"][q][family].append(v)
            if not n:
                continue
            with np.errstate(invalid="ignore"):
                by_circ = np.mean(circ[sel] > CIRC_THRESHOLD)
                by_winding = np.mean(winding[sel] > WINDING_THRESHOLD)
                by_speed = np.mean(
                    (winding[sel] > WINDING_THRESHOLD) & (speed[sel] > SPEED_THRESHOLD)
                )
                by_hole = np.mean(
                    (winding[sel] > WINDING_THRESHOLD) & (hole[sel] > HOLE_THRESHOLD)
                )
            print(
                f"{CLASS_NAMES[int(family)]:>22s} {n:6d} {med['circ']:6.3f} "
                f"{med['winding']:6.3f} {med['speed']:6.3f} {med['hole']:6.3f} | "
                f"{'':9s}{by_circ:5.2f} {by_winding:5.2f} {by_speed:5.2f} {by_hole:6.2f}"
            )
        if np.isfinite(rcr):
            # The full classifier with and without the rotating-figure rescue:
            # fraction of each static family it calls a loop (tube or rosette).
            with_rescue = res.classify(drop_runaways=False)
            without = res.classify(drop_runaways=False, rescue=False)
            loops = (OrbitShape.BOX, OrbitShape.UNCLASSIFIED)
            loop_with = ~np.isin(with_rescue.shape, loops)
            loop_without = ~np.isin(without.shape, loops)
            print(
                f"classifier: {int(with_rescue.rescued.sum())} rescued; rosettes "
                f"{int(np.sum(with_rescue.shape == OrbitShape.ROSETTE))}; loop fraction "
                f"by static family (no rescue -> rescue):"
            )
            for family in FAMILIES:
                sel = labels == family
                if sel.any():
                    print(
                        f"{CLASS_NAMES[int(family)]:>22s} "
                        f"{np.mean(loop_without[sel]):.2f} -> {np.mean(loop_with[sel]):.2f}"
                    )
            locking = values["locking"]
            collected["locking"][rcr] = locking
            with np.errstate(divide="ignore"):
                log_locking = np.log10(locking[np.isfinite(locking)])
            n_locked = int(np.sum(log_locking < -3))
            n_gap = int(np.sum((log_locking >= -4) & (log_locking < -2)))
            print(
                f"locking: {n_locked} orbits ({100 * n_locked / max(len(log_locking), 1):.2f}%) "
                f"below tol=1e-3; {n_gap} in the 1e-4..1e-2 band around it"
            )
        if not np.isfinite(rcr) or rcr == runs[-1][0]:
            collected["scatter"][rcr] = (circ, winding, labels)
    return collected


def plot(model, collected, figdir):
    """Save the three figures of one model.

    Parameters
    ----------
    model : str
        Model name.
    collected : dict
        Output of :func:`summarise`.
    figdir : str
        Output directory.
    """
    os.makedirs(figdir, exist_ok=True)
    rcr = np.array(collected["rcr"])
    # Static at the right-hand end of a log axis.
    finite = rcr[np.isfinite(rcr)]
    x = np.where(np.isfinite(rcr), rcr, 2 * finite.max() if finite.size else 1.0)
    titles = {
        "circ": r"$|\langle L_a\rangle| / \langle|L_a|\rangle$",
        "winding": r"$|\phi_\mathrm{net}| / \phi_\mathrm{range}$",
        "speed": r"$|\Omega_\phi| / \omega_R$",
        "hole": r"$\rho_\mathrm{min} / \langle r\rangle$",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained", sharex=True)
    for ax, (quantity, title) in zip(axes.flat, titles.items()):
        for family in FAMILIES:
            ax.plot(
                x,
                collected["medians"][quantity][family],
                marker="o",
                label=CLASS_NAMES[int(family)],
            )
        ax.set_xscale("log")
        if quantity == "speed":
            # Orbits whose radial motion is unresolved near corotation blow
            # the ratio up by orders of magnitude.
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel(r"$R_\mathrm{CR} / r_{1/2}$ (rightmost: static)")
    axes.flat[0].legend(fontsize="small")
    fig.suptitle(f"{model}: medians by static family")
    fig.savefig(os.path.join(figdir, f"{model}_diagnostics_vs_rcr.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(
        1,
        len(collected["scatter"]),
        figsize=(11, 5),
        layout="constrained",
        squeeze=False,
    )
    for ax, (key, (circ, winding, labels)) in zip(
        axes[0], collected["scatter"].items()
    ):
        rng = np.random.default_rng(0)
        for family in FAMILIES:
            sel = np.flatnonzero(labels == family)
            sel = rng.permutation(sel)[:3000]
            ax.scatter(
                circ[sel],
                winding[sel],
                s=2,
                alpha=0.4,
                label=CLASS_NAMES[int(family)],
                rasterized=True,
            )
        ax.set_xlabel(titles["circ"])
        ax.set_ylabel(titles["winding"])
        ax.set_title("static" if not np.isfinite(key) else f"R_CR = {key:g}")
    axes[0, 0].legend(markerscale=4, fontsize="small")
    fig.savefig(os.path.join(figdir, f"{model}_circ_vs_winding.png"), dpi=200)
    plt.close(fig)

    if collected["locking"]:
        fig, ax = plt.subplots(layout="constrained")
        bins = np.linspace(-9, 2, 111)
        for key, locking in collected["locking"].items():
            with np.errstate(divide="ignore"):
                values = np.log10(locking[np.isfinite(locking) & (locking > 0)])
            ax.hist(values, bins=bins, histtype="step", label=f"R_CR = {key:g}")
        ax.axvline(-3, color="k", ls="--", lw=1, label="tol = 1e-3")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\log_{10} |\Omega_{\phi,\mathrm{inertial}}| / |\Omega_p|$")
        ax.legend(fontsize="small")
        ax.set_title(f"{model}: pattern locking")
        fig.savefig(os.path.join(figdir, f"{model}_locking.png"), dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grid-dir", type=str, required=True)
    ap.add_argument("--figdir", type=str, default=None, help="save figures here")
    ap.add_argument("--models", nargs="+", default=None)
    args = ap.parse_args()
    lf.set_verbosity("WARNING")
    runs = find_runs(args.grid_dir)
    for model in args.models or sorted(runs):
        collected = summarise(model, runs[model])
        if args.figdir:
            plot(model, collected, args.figdir)
            print(f"figures for {model} saved to {args.figdir}")


if __name__ == "__main__":
    main()
