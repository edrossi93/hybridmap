"""Supplementary Figures S1, S2 and S4, rebuilt from the deposited tables.

The three of them were still the artwork submitted with the first version, and
their labels had not followed the paper. S1's legend read "basalt-mixed" and
"linen", S2's panel titles called the load-controlled map a "3D acquisition", and S4
carried "basalt mixed" in its legend and "mixed-mode" in a panel title. All three
names were retired in this revision, and the reply tells the reviewer that the
class is called mixed-volume "in the manuscript, the supplement and the figures".
Nothing had read the figures, because the names live inside a raster.

Rebuilding them from the deposit rather than editing the pixels also puts them
under the same reproducibility as the paper's own figures, which is where they
should have been: until now they were the only figures in the package that no
script could produce.

Every number the captions and the Discussion quote is recomputed here and
printed, so a disagreement shows up rather than being drawn.
"""

import sys
from pathlib import Path

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import clustering_check as cc
import figstyle as fs

DATA = Path(__file__).resolve().parent.parent / "data"

ORDER = ["basalt", "mixed-volume", "flax", "matrix", "QC/invalid"]
E_LOWER, E_UPPER_SHALLOW = 15.0, 70.0
KNN = 8
EDGE_UM = 6.0     # a band at each end of the zone, one grid pitch wide


def depth_resolved():
    d = pd.read_csv(DATA / "classified_depth_resolved.csv")
    return d


def s1(outdir):
    """The rule against the unsupervised partition, on the same indents."""
    d = depth_resolved()
    fs.apply()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(fs.ONE_HALF, 1.02 * fs.ONE_HALF),
        layout="constrained")

    colours = {**fs.CONSTITUENT}
    for ax, col, title, cmapping in (
            (ax1, "constituent", "Rule-based classes", colours),
            (ax2, "kmeans_k4", "k-means, k = 4", None)):
        if cmapping:
            for name in ORDER:
                m = d[col] == name
                ax.scatter(d.x_um[m], d.y_um[m], s=7, marker="s",
                           color=cmapping[name if name != "QC/invalid" else "qc"],
                           label=name, linewidths=0)
        else:
            palette = [fs.WONG["blue"], fs.WONG["orange"], fs.WONG["green"],
                       fs.WONG["red"]]
            for k in range(4):
                m = d[col] == k
                ax.scatter(d.x_um[m], d.y_um[m], s=7, marker="s",
                           color=palette[k], label=f"cluster {k}", linewidths=0)
        ax.set_xlabel("x (µm)")
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=7.0,
                  markerscale=1.5, handletextpad=0.3, frameon=False, ncol=2)
    ax1.set_ylabel("y (µm)")

    # The agreement is the deposit's own, not a second implementation of it:
    # the paper quotes clustering_check, so the figure must quote the same run.
    r = cc.run(d)
    ari, ari_star = r["ARI"], r["ARI_star"]
    fs.panel_label(ax1, "a", dx=-0.30, dy=1.10)
    fs.panel_label(ax2, "b", dx=-0.30, dy=1.10)
    fs.save(fig, "FigS5_rule_vs_kmeans", outdir)
    # ARI* merges the compliant pair on both sides, the two lowest clusters and
    # flax with matrix. The caption printed 0.65 for two rounds, which was the
    # one-sided merge; on the two-sided comparison it describes, 0.83.
    print(f"  S1: ARI {ari:.3f}, ARI* after merging the compliant pair on "
          f"both sides {ari_star:.3f}  (the caption prints 0.40 and, "
          f"corrected, 0.83)")


def s2(outdir):
    """Where the thresholds sit in the modulus distribution of each zone."""
    d = pd.read_csv(DATA / "classified_shallow.csv")
    zones = [("basalt", "Basalt-rich zone", True),
             ("flax", "Flax-rich zone", False)]
    fs.apply()
    fig, axes = plt.subplots(1, 2, figsize=(fs.DOUBLE, 0.42 * fs.DOUBLE),
                             layout="constrained")
    for ax, (zone, title, logy) in zip(axes, zones):
        e = d.loc[d.zone == zone, "E_GPa"].dropna()
        # Both panels run to 130 GPa and both are logarithmic in count.
        # Reviewer 3 pointed out that the flax panel used to stop at 25 GPa,
        # which hid the 492 indents above 15 GPa: on a linear count axis they
        # are invisible beside the compliant peak, and on a truncated modulus
        # axis they are not drawn at all. The stiff tail of the flax-rich zone
        # is the mixed-volume population, and it is the point of the figure.
        top, logy = 130, True
        ax.hist(e[e <= top], bins=60, color="0.72", edgecolor="k",
                linewidth=0.5)
        ax.axvline(E_LOWER, color=fs.PAPER["mixed-volume"], lw=1.8,
                   ls=(0, (5, 2)), label="15 GPa, compliant / mixed-volume")
        ax.set_yscale("log")
        ax.axvline(E_UPPER_SHALLOW, color=fs.PAPER["basalt"], lw=1.8,
                   ls=(0, (5, 2)),
                   label="70 GPa, mixed-volume / pure-axial basalt")
        ax.set_xlim(0, top)
        ax.set_xlabel("Indentation modulus, E (GPa)")
        ax.set_ylabel("Count")
        ax.set_title(f"{title}, load-controlled map (n = {len(d[d.zone == zone])})",
                     fontsize=8.6)
        ax.legend(loc="upper right", fontsize=7.2, handlelength=1.6)
        ax.minorticks_on()
    fs.panel_label(axes[0], "a", dx=-0.16, dy=1.12)
    fs.panel_label(axes[1], "b", dx=-0.16, dy=1.12)
    fs.save(fig, "FigS3_threshold_histograms", outdir)
    for zone, *_ in zones:
        sub = d[d.zone == zone]
        print(f"  S2: {zone}-rich zone n = {len(sub)}, "
              f"{int((sub.E_GPa < 1).sum())} below 1 GPa, "
              f"{int((sub.E_GPa > 15).sum())} above 15 GPa")


def s3(outdir):
    """Why k = 4 is a choice and not a diagnosis.

    The same three curves the manuscript describes: the sum of squares, the
    silhouette and the Gaussian-mixture BIC, over k = 2 to 8, on the same nine
    features and the same quality-controlled indents as everything else. It was
    the one supplementary figure with no stale label, and the one no script
    could produce.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    d = depth_resolved()
    keep = (d.constituent != "QC/invalid") & d[cc.FEATURES].notna().all(axis=1)
    X = StandardScaler().fit_transform(d.loc[keep, cc.FEATURES].to_numpy())

    ks = list(range(2, 9))
    inertia, sil, bic = [], [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=cc.N_INIT,
                    random_state=cc.SEED).fit(X)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X, km.labels_))
        bic.append(GaussianMixture(n_components=k, covariance_type="full",
                                   n_init=10, random_state=cc.SEED)
                   .fit(X).bic(X))

    fs.apply()
    fig, axes = plt.subplots(1, 3, figsize=(fs.DOUBLE, 0.40 * fs.DOUBLE),
                             layout="constrained")
    for ax, y, label, title in (
            (axes[0], inertia, "Inertia", "k-means sum of squares"),
            (axes[1], sil, "Silhouette", "Silhouette score"),
            (axes[2], bic, "BIC", "Gaussian-mixture BIC")):
        ax.plot(ks, y, "o-", color=fs.WONG["orange"], ms=5, mfc="white",
                mew=1.4)
        ax.axvline(4, color="0.55", lw=0.9, ls=(0, (2, 3)))
        ax.text(4.12, ax.get_ylim()[1], " k = 4", fontsize=7.6, va="top")
        ax.set_xlabel("k (clusters)")
        ax.set_ylabel(label)
        ax.set_title(title, fontsize=8.6)
        ax.minorticks_on()
    for ax, letter in zip(axes, "abc"):
        fs.panel_label(ax, letter, dx=-0.30, dy=1.16)
    fs.save(fig, "FigS4_cluster_diagnostics", outdir)
    print(f"  S3: silhouette peaks at k = {ks[int(np.argmax(sil))]}, "
          f"BIC falls monotonically: {all(np.diff(bic) < 0)}")


def neighbour_fraction(sub, pool, k):
    """Fraction of each indent's k nearest neighbours that are pure basalt.

    The neighbours are drawn from every indent in the zone, including those the
    quality filter rejected. A rejected indent still occupies its place on the
    grid, and leaving it out would move its neighbours' neighbours. Using only
    the classified indents instead moves p at k = 8 from the published
    2.3e-12 to 4.9e-10, on the quality mask deposited now; on the superseded
    mask the same pair read 8.5e-11 and 7.4e-11, which is how the published
    figure was identified.
    """
    xy = sub[["x_um", "y_um"]].to_numpy()
    pxy = pool[["x_um", "y_um"]].to_numpy()
    dist = np.linalg.norm(xy[:, None, :] - pxy[None, :, :], axis=2)
    for i, (a, b) in enumerate(xy):
        dist[i, np.where((pxy[:, 0] == a) & (pxy[:, 1] == b))[0]] = np.inf
    idx = np.argsort(dist, axis=1)[:, :k]
    is_basalt = (pool.constituent.to_numpy() == "basalt").astype(float)
    return is_basalt[idx].mean(axis=1)


def s4(outdir):
    """Are mixed-volume contacts where the two phases meet?"""
    d = depth_resolved()
    zone = d[d.zone == "basalt-centre"]
    sub = zone[zone.constituent != "QC/invalid"].copy()
    qc = zone[zone.constituent == "QC/invalid"]

    fs.apply()
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(fs.DOUBLE, 0.46 * fs.DOUBLE),
        gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]}, layout="constrained")
    fig.get_layout_engine().set(wspace=0.09)

    ax1.scatter(qc.y_um, qc.x_um, s=9, color="0.85", linewidths=0,
                label=f"QC/invalid (n = {len(qc)})")
    for name, colour in (("mixed-volume", fs.PAPER["mixed-volume"]),
                         ("basalt", fs.PAPER["basalt"]),
                         ("matrix", fs.PAPER["matrix"])):
        m = sub.constituent == name
        ax1.scatter(sub.y_um[m], sub.x_um[m], s=9, color=colour, linewidths=0,
                    label=f"{name} (n = {int(m.sum())})")
    ax1.set_xlabel("y position (µm)")
    ax1.set_ylabel("x position (µm)")
    ax1.set_title("Classes by position, basalt-centre zone", fontsize=8.0)
    ax1.legend(fontsize=6.8, frameon=True, framealpha=0.9, loc="upper right")

    ks = list(range(3, 21))
    ps = []
    for k in ks:
        f = neighbour_fraction(sub, zone, k)
        a = f[sub.constituent.to_numpy() == "mixed-volume"]
        b = f[sub.constituent.to_numpy() == "matrix"]
        ps.append(stats.mannwhitneyu(a, b, alternative="greater").pvalue)
    ax2.plot(ks, -np.log10(ps), "o-", color=fs.WONG["orange"], ms=4)
    for y, lab in ((3, "p = 0.001"), (-np.log10(0.05), "p = 0.05")):
        ax2.axhline(y, color="0.6", lw=0.8, ls=(0, (4, 3)))
        ax2.text(20, y + 0.2, lab, fontsize=7, ha="right", color="0.35")
    ax2.set_xlabel("k (nearest neighbours)")
    ax2.set_ylabel("$-\\log_{10}(p)$, Mann–Whitney U")
    ax2.set_title("Basalt neighbours,\nmixed against matrix", fontsize=8.0)
    ax2.minorticks_on()

    f8 = neighbour_fraction(sub, zone, KNN)
    groups, labels = [], []
    for name in ("basalt", "mixed-volume", "matrix"):
        m = sub.constituent.to_numpy() == name
        groups.append(f8[m])
        labels.append(f"{name.replace(chr(45), chr(45) + chr(10))}\n{int(m.sum())}")
    bp = ax3.boxplot(groups, patch_artist=True, widths=0.6, tick_labels=labels)
    for patch, name in zip(bp["boxes"], ("basalt", "mixed-volume", "matrix")):
        patch.set_facecolor(fs.PAPER[name])
        patch.set_alpha(0.85)
    for line in bp["medians"]:
        line.set_color("k")
    ax3.set_ylabel(f"Basalt fraction, {KNN} neighbours", fontsize=8.5)
    ax3.set_title(f"Neighbourhood at k = {KNN}", fontsize=8.0)
    ax3.tick_params(axis="x", labelsize=6.8)

    p8 = ps[ks.index(KNN)]
    stats_out = spatial_statistics(d, sub, zone, f8, ps, ks)
    for ax, letter, dx in ((ax1, "a", -0.14), (ax2, "b", -0.30), (ax3, "c", -0.24)):
        fs.panel_label(ax, letter, dx=dx, dy=1.14)
    fs.save(fig, "FigS6_mixed_volume_localisation", outdir)
    print(f"  S4: counts basalt {int((sub.constituent=='basalt').sum())}, "
          f"mixed-volume {int((sub.constituent=='mixed-volume').sum())}, "
          f"matrix {int((sub.constituent=='matrix').sum())}, QC {len(qc)}")
    print(f"  S4: p at k = {KNN} is {p8:.2e}; largest p over k = 3 to 20 is "
          f"{max(ps):.1e}  (the paper says 2.3e-12 and all below 1e-9)")


def spatial_statistics(d, sub, zone, f8, ps, ks):
    """Every spatial claim §4.1 makes, computed once and written down.

    The manuscript quotes four of these and none of them could be reproduced
    from the deposit before: the p at k = 8, the stability across k, the p after
    excluding a band at each edge of the zone, and the p on the load-controlled map.
    The definitions are here so that the numbers mean something a reader can
    check, rather than being reported and taken on trust.
    """
    rng = np.random.default_rng(42)
    c = sub.constituent.to_numpy()
    med = {n: float(np.median(f8[c == n]))
           for n in ("basalt", "mixed-volume", "matrix")}
    a, b = f8[c == "mixed-volume"], f8[c == "matrix"]
    draws = [np.median(a[rng.integers(0, len(a), len(a))])
             - np.median(b[rng.integers(0, len(b), len(b))]) for _ in range(4000)]

    # A band of one grid pitch at each end of the zone along its long axis, so
    # that no indent counted here can have neighbours outside the zone.
    lo, hi = zone.y_um.min(), zone.y_um.max()
    inner = sub[(sub.y_um >= lo + EDGE_UM) & (sub.y_um <= hi - EDGE_UM)]
    fi = neighbour_fraction(inner, zone, KNN)
    ci = inner.constituent.to_numpy()
    p_edge = stats.mannwhitneyu(fi[ci == "mixed-volume"], fi[ci == "matrix"],
                                alternative="greater").pvalue

    s = pd.read_csv(DATA / "classified_shallow.csv")
    szone = s[s.zone == "basalt"]
    sclean = szone[szone.constituent != "QC/invalid"]
    fsh = neighbour_fraction(sclean, szone, KNN)
    cs = sclean.constituent.to_numpy()
    p_shallow = stats.mannwhitneyu(fsh[cs == "mixed-volume"], fsh[cs == "matrix"],
                                   alternative="greater").pvalue

    out = {
        "neighbours": KNN,
        "pool": "every indent in the zone, including quality-rejected ones",
        "test": "Mann-Whitney U, one-sided, mixed-volume above matrix",
        "median_basalt_neighbour_fraction": {k: round(v, 4) for k, v in med.items()},
        "median_difference_ci95": [round(float(np.percentile(draws, 2.5)), 4),
                                   round(float(np.percentile(draws, 97.5)), 4)],
        "p_depth_resolved": float(ps[ks.index(KNN)]),
        "p_max_over_k_3_to_20": float(max(ps)),
        "edge_band_um": EDGE_UM,
        "p_excluding_edge_bands": float(p_edge),
        "n_after_edge_exclusion": int(len(inner)),
        "p_load_controlled_map": float(p_shallow),
        "n_load_controlled_zone": int(len(szone)),
    }
    (DATA / "spatial_localisation.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"  S4: medians {out['median_basalt_neighbour_fraction']}, "
          f"CI {out['median_difference_ci95']}")
    print(f"  S4: p after excluding {EDGE_UM} µm at each end "
          f"{p_edge:.2e} (n = {len(inner)}); on the load-controlled map "
          f"{p_shallow:.2e} (n = {len(szone)})")
    return out


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    s1(out)
    s2(out)
    s3(out)
    s4(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
