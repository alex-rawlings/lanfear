"""Help choose ``diffusion_threshold`` and ``diffusion_drop``.

Orbits whose Laskar frequency-diffusion rate exceeds ``diffusion_threshold`` are
labelled irregular (chaotic). If the run used extension (``max_extensions`` in
``analyse_family``), that threshold also *screens* which orbits are
re-integrated for longer, and an extended orbit is chaotic only if its rate did
not fall by at least ``diffusion_drop`` on extension. The right values depend on
the integration length, the potential and the orbit population, so this script
inspects a saved orbit file and shows what each candidate does.

Threshold (always):

* the distribution of ``log10(rate)`` at the *base* integration length (an
  extended orbit is represented by its rate before extension, so orbits of
  different lengths are not mixed);
* per candidate threshold, how many orbits are flagged (with extension, this is
  the fraction that would be re-integrated, i.e. the cost) and which regular
  families they are taken from -- a threshold that flags many tubes is too low,
  since tubes are the most regular orbits;
* two suggestions: the Otsu split of the log-rate histogram, and the threshold
  that best agrees with the *independent* spectral irregular test (``>3`` base
  frequencies, Frigo et al. 2021), scored by F1. Both are heuristics.

Drop factor (only for results with extended orbits):

* the distribution of ``rate_after / rate_before`` over the extended orbits.
  Regular orbits fall (roughly as the length ratio to the power -2) and chaotic
  ones do not, so this is far more clearly bimodal than the raw rate;
* per candidate drop factor, the fraction of extended orbits judged chaotic and
  which regular families they are taken from;
* an Otsu suggestion for the drop factor.

    python scripts/choose_diffusion_threshold.py orbits.npz
    python scripts/choose_diffusion_threshold.py orbits.npz --plot threshold.png

Then use the values you settle on, in ``analyse_family`` (to screen) and in
``classify`` (to label):

    res = lf.analyse_family(..., max_extensions=2,
                            diffusion_threshold=0.03, diffusion_drop=0.5)
    cls = res.classify(diffusion_threshold=0.03, diffusion_drop=0.5)

If a histogram is not bimodal, no value cleanly separates two populations and
the script says so.
"""

import argparse
import numpy as np
import lanfear as lf


def otsu_threshold(log_rate, bins=64):
    """Otsu's threshold of a 1-D sample: maximises the between-class variance.

    Parameters
    ----------
    log_rate : numpy.ndarray
        Finite ``log10`` diffusion rates.
    bins : int, optional
        Histogram bins used to search for the split.

    Returns
    -------
    threshold : float
        Suggested split, in ``log10`` units.
    separation : float
        Between-class variance as a fraction of the total variance, in [0, 1].
        A clearly bimodal sample gives ~0.8 or more; a unimodal one gives ~0.64
        (a Gaussian), so values below ~0.75 signal a weak split.
    """
    hist, edges = np.histogram(log_rate, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    weight = hist.astype(float)
    total = weight.sum()
    cum_w = np.cumsum(weight)
    cum_wx = np.cumsum(weight * centres)
    w0 = cum_w
    w1 = total - cum_w
    m0 = cum_wx / np.maximum(w0, 1e-30)
    m1 = (cum_wx[-1] - cum_wx) / np.maximum(w1, 1e-30)
    between = w0 * w1 * (m0 - m1) ** 2 / total**2  # between-class variance
    best = int(np.argmax(between[:-1]))
    mean = cum_wx[-1] / total
    variance = np.sum(weight * (centres - mean) ** 2) / total
    separation = between[best] / variance if variance > 0 else 0.0
    return float(edges[best + 1]), float(separation)


def best_agreement_threshold(rate, spectral_irregular, candidates):
    """Threshold whose flagged set best matches the spectral irregular test.

    Parameters
    ----------
    rate : numpy.ndarray
        (N,) diffusion rate of the successfully integrated orbits.
    spectral_irregular : numpy.ndarray
        (N,) boolean, True where the spectral criterion says irregular.
    candidates : numpy.ndarray
        Thresholds to score.

    Returns
    -------
    threshold : float
        Candidate with the highest F1 score (NaN if the spectral test flags
        nothing, so there is nothing to agree with).
    f1 : float
        The corresponding F1 score.
    """
    if not spectral_irregular.any():
        return float("nan"), 0.0
    best_f1, best_t = -1.0, float("nan")
    for t in candidates:
        flagged = rate > t
        true_pos = np.sum(flagged & spectral_irregular)
        denom = flagged.sum() + spectral_irregular.sum()
        f1 = 2.0 * true_pos / denom if denom else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, float(best_f1)


def plot_summary(
    rate,
    spectral_irregular,
    thresholds,
    suggestions,
    path,
    ratio=None,
    drops=(),
    drop_suggestion=None,
):
    """Histograms of log10 rate (and, if given, of the extension rate ratio).

    Parameters
    ----------
    rate : numpy.ndarray
        (N,) finite, positive base-length diffusion rates.
    spectral_irregular : numpy.ndarray
        (N,) boolean spectral irregular flag.
    thresholds : sequence of float
        Candidate thresholds, drawn as faint vertical lines.
    suggestions : dict
        ``label -> threshold`` drawn as bold labelled lines.
    path : str
        Output image path.
    ratio : numpy.ndarray, optional
        Rate ratios ``after / before`` of the extended orbits; adds a second
        panel.
    drops : sequence of float, optional
        Candidate drop factors, drawn as faint lines on the ratio panel.
    drop_suggestion : float, optional
        Suggested drop factor, drawn as a bold line on the ratio panel.
    """
    import matplotlib.pyplot as plt

    n_panels = 2 if ratio is not None and len(ratio) else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 4), squeeze=False)
    ax = axes[0, 0]
    log_rate = np.log10(rate)
    edges = np.linspace(log_rate.min(), log_rate.max(), 61)
    ax.hist(
        log_rate[~spectral_irregular], bins=edges, alpha=0.7, label="spectrally regular"
    )
    if spectral_irregular.any():
        ax.hist(
            log_rate[spectral_irregular],
            bins=edges,
            alpha=0.7,
            label="spectrally irregular",
        )
    for t in thresholds:
        ax.axvline(np.log10(t), color="grey", lw=0.5, ls=":")
    for i, (label, t) in enumerate(suggestions.items()):
        if np.isfinite(t):
            ax.axvline(
                np.log10(t), color=f"C{i + 2}", lw=1.5, label=f"{label}: {t:.2g}"
            )
    ax.set_xlabel(r"$\log_{10}$ diffusion rate (base length)")
    ax.set_ylabel("orbits")
    ax.legend()

    if n_panels == 2:
        ax = axes[0, 1]
        log_ratio = np.log10(ratio)
        ax.hist(log_ratio, bins=40, color="C4", alpha=0.7)
        for d in drops:
            ax.axvline(np.log10(d), color="grey", lw=0.5, ls=":")
        if drop_suggestion is not None and np.isfinite(drop_suggestion):
            ax.axvline(
                np.log10(drop_suggestion),
                color="C3",
                lw=1.5,
                label=f"Otsu: {drop_suggestion:.2g}",
            )
            ax.legend()
        ax.set_xlabel(r"$\log_{10}$ (rate after / rate before extension)")
        ax.set_ylabel("extended orbits")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"Saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("file", type=str, help="OrbitResults .npz file (from res.save)")
    ap.add_argument(
        "--amp-frac",
        type=float,
        default=0.05,
        help="axes weaker than this fraction of the strongest are ignored",
    )
    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0],
        help="candidate thresholds to tabulate",
    )
    ap.add_argument(
        "--drops",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5, 0.75, 1.0],
        help="candidate drop factors to tabulate (extended results only)",
    )
    ap.add_argument("--plot", type=str, default=None, help="save a summary figure")
    args = ap.parse_args()

    res = lf.OrbitResults.load(args.file)

    # Classify with the cut disabled: families are then the regular ones, and the
    # IRREGULAR label is the independent spectral criterion.
    cls = res.classify(diffusion_threshold=None)
    final_rate = res.diffusion_rate(amp_frac=args.amp_frac)
    previous = res.diffusion_previous
    extended = (
        np.isfinite(previous)
        if previous is not None
        else np.zeros(len(final_rate), bool)
    )
    # Base-length rate: an extended orbit's rate before its extension, so all
    # orbits are compared at the same integration length. Exact when each orbit
    # was extended once; for repeated extensions it is the last pre-extension
    # rate (a longer window than the base).
    rate_all = np.where(extended, previous, final_rate)
    use = res.ok & np.isfinite(rate_all) & (rate_all > 0)
    rate = rate_all[use]
    labels = cls.labels[use]
    spectral = labels == lf.OrbitClass.IRREGULAR
    n = len(rate)
    if n < 20:
        raise SystemExit(f"Only {n} orbits with a usable rate; too few to judge.")
    log_rate = np.log10(rate)

    skipped = int(res.ok.sum()) - n
    print(
        f"{n} orbits with a measurable diffusion rate "
        f"({skipped} ok orbits skipped: NaN or zero rate)"
    )
    if extended.any():
        print(
            f"{int(extended.sum())} orbits were extended; the threshold analysis "
            f"below uses their rates from before extension"
        )
        if res.n_periods_used is not None and len(np.unique(res.n_periods_used)) > 2:
            print(
                "  NOTE: some orbits were extended more than once, so their "
                "'base' rate is from a longer window than the rest."
            )
    pct = np.percentile(log_rate, [5, 25, 50, 75, 95])
    print(
        "log10(rate) percentiles 5/25/50/75/95: " + "  ".join(f"{p:+.2f}" for p in pct)
    )
    print(
        f"spectral irregular test flags {int(spectral.sum())} "
        f"({100 * spectral.mean():.1f}%)\n"
    )

    # Fraction of each family removed at each candidate threshold.
    families = [int(c) for c in np.unique(labels) if c != lf.OrbitClass.IRREGULAR]
    names = [cls.class_names[c] for c in families]
    head = f"{'threshold':>10} {'flagged':>8}"
    for name in names:
        head += f" {name[:13]:>14}"
    print(head)
    print("(family columns: % of that regular family the threshold would flag)")
    for t in sorted(args.thresholds):
        flagged = rate > t
        row = f"{t:>10.1e} {100 * flagged.mean():>7.1f}%"
        for c in families:
            row += f" {100 * flagged[labels == c].mean():>13.1f}%"
        print(row)

    # Automatic suggestions.
    otsu_log, separation = otsu_threshold(log_rate)
    otsu = 10.0**otsu_log
    grid = 10.0 ** np.linspace(log_rate.min(), log_rate.max(), 200)
    agree, f1 = best_agreement_threshold(rate, spectral, grid)

    print("\nSuggestions")
    print(f"  Otsu split of log10(rate):    {otsu:.2g}   (separation {separation:.2f})")
    if np.isfinite(agree):
        print(f"  best match to spectral test:  {agree:.2g}   (F1 {f1:.2f})")
    else:
        print("  best match to spectral test:  n/a (spectral test flagged nothing)")
    if separation < 0.75:
        print(
            "  WARNING: the rate distribution is not clearly bimodal, so the Otsu "
            "split is weakly determined. Consider a longer integration or a "
            "conservative threshold such as 0.1."
        )
    if np.isfinite(agree) and abs(np.log10(agree) - otsu_log) > 1.0:
        print(
            "  NOTE: the two suggestions differ by more than a decade; inspect "
            "the histogram (--plot) before choosing."
        )

    # Drop factor: the rate ratio over the extended orbits.
    ratio = drop_otsu = None
    good = extended & res.ok & np.isfinite(final_rate) & (final_rate > 0)
    good &= np.isfinite(previous) & (previous > 0) if previous is not None else good
    if good.any():
        ratio = final_rate[good] / previous[good]
        ext_labels = cls.labels[good]
        log_ratio = np.log10(ratio)
        print(f"\nDrop factor: rate after / before extension, {len(ratio)} orbits")
        pct = np.percentile(log_ratio, [10, 25, 50, 75, 90])
        print(
            "  ratio percentiles 10/25/50/75/90: "
            + "  ".join(f"{10.0**p:.3g}" for p in pct)
        )
        if res.n_periods_used is not None:
            top = int(res.n_periods_used[good].max())
            print(
                f"  (a regular orbit's rate falls roughly as the length ratio to "
                f"the power -2; e.g. ~0.25 for a doubling. Longest run: {top} "
                f"periods, base {res.n_periods}.)"
            )
        fam = [int(c) for c in np.unique(ext_labels) if c != lf.OrbitClass.IRREGULAR]
        head = f"{'drop':>10} {'chaotic':>8}"
        for c in fam:
            head += f" {cls.class_names[c][:13]:>14}"
        print(head)
        print("(family columns: % of that regular family judged chaotic)")
        for d in sorted(args.drops):
            chaotic = ratio >= d
            row = f"{d:>10.2f} {100 * chaotic.mean():>7.1f}%"
            for c in fam:
                row += f" {100 * chaotic[ext_labels == c].mean():>13.1f}%"
            print(row)
        if len(ratio) >= 20:
            drop_log, drop_sep = otsu_threshold(log_ratio)
            drop_otsu = 10.0**drop_log
            print(
                f"  Otsu split of log10(ratio): {drop_otsu:.2g}   "
                f"(separation {drop_sep:.2f})"
            )
            if drop_sep < 0.75:
                print(
                    "  WARNING: the ratio distribution is not clearly bimodal; "
                    "extension did not cleanly separate falling from "
                    "non-falling drift."
                )
        else:
            print("  Too few extended orbits for a suggestion.")

    print(
        "\nApply with:  res.classify(diffusion_threshold=<value>, "
        "diffusion_drop=<value>)  (threshold None disables the cut)"
    )

    if args.plot:
        plot_summary(
            rate,
            spectral,
            args.thresholds,
            {"Otsu": otsu, "spectral match": agree},
            args.plot,
            ratio=ratio,
            drops=args.drops,
            drop_suggestion=drop_otsu,
        )


if __name__ == "__main__":
    main()
