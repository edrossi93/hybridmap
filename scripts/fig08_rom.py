"""Figure 8 - the rule-of-mixtures construction, within zone and at the laminate scale.

Panel (a) works inside the depth-resolved strip, which is banded into three
architectural zones (flax-rich bottom, basalt-rich centre, flax-rich top). For
each zone the within-zone class fractions and the class median moduli at
z = 150 nm give Voigt, Hill and Reuss. A fourth bar shows the shallow-corrected
Hill bound, in which the basalt input is replaced by the 86 GPa median measured
in the shallow map, where the fibres are not depressed by substrate compliance.
The zone-averaged modulus actually measured is marked so the reader can see
which bound lands closest.

Panel (b) is the laminate-scale construction from the shallow map. The two
measured flexural moduli are drawn across it with their standard deviations,
and the gap to the Voigt bound is annotated as a fraction of the bound, the
same definition used in the text.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs
from rule_of_mixtures import bounds, class_table, CLASSES

DATA = Path(__file__).resolve().parent.parent / "data"

# median modulus of the pure-axial basalt class in the shallow acquisition
BASALT_SHALLOW_GPA = 86.0

# Tabulated-input baselines quoted in §3.4, using tabulated constituents and the
# measured fibre volume fraction of 0.649.
TABULATED_VOIGT = 35.75
TABULATED_HT = 27.1

ZONE_ORDER = ["flax-bottom", "basalt-centre", "flax-top"]
# kept short: three category names under a narrow panel run into one another
ZONE_LABEL = {"flax-bottom": "bottom", "basalt-centre": "centre",
              "flax-top": "top"}


def zone_table(depth, modulus_col="E_150"):
    """Per-zone bounds inside the depth-resolved strip.

    Only the classes actually present in a zone enter its construction; a class
    with no indents there has no median and must not be given one.
    """
    rows = []
    d = depth[depth.constituent.isin(CLASSES)]
    for zone in ZONE_ORDER:
        sub = d[d["zone"] == zone]
        present = [c for c in CLASSES if (sub.constituent == c).sum() > 0]
        n = np.array([(sub.constituent == c).sum() for c in present], float)
        E = np.array([sub.loc[sub.constituent == c, modulus_col].median()
                      for c in present])
        v, h, r = bounds(n / n.sum(), E)

        E_corr = E.copy()
        if "basalt" in present:
            E_corr[present.index("basalt")] = BASALT_SHALLOW_GPA
        v_corr, h_corr, _ = bounds(n / n.sum(), E_corr)

        rows.append({"zone": zone, "n": int(n.sum()),
                     "voigt": v, "hill": h, "reuss": r,
                     "voigt_shallow_corrected": v_corr,
                     "hill_shallow_corrected": h_corr,
                     "observed": sub[modulus_col].mean()})
    return pd.DataFrame(rows)


def build(outdir):
    # Created here rather than assumed. Run from a fresh checkout with an
    # output directory that does not exist yet, these scripts used to fail
    # on the savefig at the end, after doing all the work.
    Path(outdir).mkdir(parents=True, exist_ok=True)
    depth = pd.read_csv(DATA / "classified_depth_resolved.csv")
    shallow = pd.read_csv(DATA / "classified_shallow.csv")
    moduli = pd.read_csv(DATA / "flexural_moduli.csv")

    zt = zone_table(depth)
    tab = class_table(shallow)
    v_lam, h_lam, r_lam = bounds(tab.fraction, tab.E_median_GPa)
    meas = (moduli.groupby("configuration")["E_D790_toe_compensated_GPa"]
            .agg(["mean", "std"]))

    fs.set_style("bold", font_scale=0.78)

    # Axes placed explicitly. Constrained layout packs the canvas edge to edge,
    # which leaves no room outside the axes for a panel letter: it then lands on
    # the tick labels or on the axis label.
    W = 6.30
    # panel (a) is given more width by pulling the left margin in to the
    # least that still holds its axis label, tick labels and letter
    LEFT, GAP, RIGHT, TOP = 0.80, 0.70, 0.10, 0.34
    PANEL_H, BOTTOM = 2.20, 1.72
    A_W = (W - LEFT - GAP - RIGHT) * 0.48
    B_W = (W - LEFT - GAP - RIGHT) * 0.52
    H = TOP + PANEL_H + BOTTOM
    fig = plt.figure(figsize=(W, H))
    ax_a = fig.add_axes([LEFT / W, BOTTOM / H, A_W / W, PANEL_H / H])
    ax_b = fig.add_axes([(LEFT + A_W + GAP) / W, BOTTOM / H, B_W / W, PANEL_H / H])

    # ---- panel a ----------------------------------------------------------
    # solid bars use the depth-resolved basalt median; hatched bars swap in the
    # shallow-map basalt value, which is the comparison the text discusses
    series = [("voigt", "Voigt", fs.PAPER["voigt"], ""),
              ("hill", "Hill", fs.PAPER["hill"], ""),
              ("reuss", "Reuss", fs.PAPER["reuss"], ""),
              ("hill_shallow_corrected", "Hill (shallow-corr.)", fs.PAPER["corrected"], ""),
              ("observed", "Observed", fs.PAPER["observed"], "")]
    x = np.arange(len(zt))
    w = 0.16
    for k, (key, name, c, hatch) in enumerate(series):
        ax_a.bar(x + (k - 2) * w, zt[key], w, color=c, edgecolor="k",
                 linewidth=1.1, hatch=hatch, label=name, zorder=3)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([ZONE_LABEL[z] for z in zt.zone])
    ax_a.set_xlabel("Architectural zone")   # composition given in the caption
    ax_a.set_ylabel(fs.AXIS["E"])   # the depth is stated in the caption
    ax_a.set_ylim(0, 58)
    ax_a.legend(loc="upper left", handlelength=1.3, fontsize=8.5, frameon=False,
                handletextpad=0.6, borderpad=0.4)
    ax_a.minorticks_on()
    ax_a.tick_params(axis="x", which="minor", bottom=False, top=False)

    # ---- panel b ----------------------------------------------------------
    # Vertical bars, as submitted. Reviewer 3 comment 22 asked only for the
    # tabulated-input Voigt to be added beside the measured-input one, so the two
    # tabulated baselines are inserted and nothing else about the panel changes.
    # The bars are labelled with the term the paper uses for that construction,
    # a Voigt sum built from tabulated constituent moduli, so that the artwork
    # and the text cannot drift apart again.
    # Hatched bars are predictions, solid bars are measurements.
    labels = ["Voigt", "Hill", "Reuss", "Voigt (tabulated inputs)",
              "Halpin-Tsai (tabulated inputs)", "Basalt-long.", "Flax-long."]
    values = [v_lam, h_lam, r_lam, TABULATED_VOIGT, TABULATED_HT,
              meas.loc["basalt_face", "mean"], meas.loc["flax_face", "mean"]]
    errs = [0, 0, 0, 0, 0,
            meas.loc["basalt_face", "std"], meas.loc["flax_face", "std"]]
    cols = [fs.PAPER["voigt"], fs.PAPER["hill"], fs.PAPER["reuss"],
            fs.PAPER["corrected"], fs.PAPER["corrected"],
            fs.PAPER["basalt"], fs.PAPER["flax"]]
    hatch = ["//", "//", "//", "//", "//", "", ""]

    xb = np.arange(len(labels))
    ax_b.bar(xb, values, 0.66, yerr=errs, capsize=4, color=cols, edgecolor="k",
             linewidth=1.1, zorder=3, error_kw=dict(lw=1.4))
    for patch, h in zip(ax_b.patches, hatch):
        patch.set_hatch(h)
    for xi, val, e in zip(xb, values, errs):
        ax_b.text(xi, val + e + 0.8, f"{val:.1f}", ha="center", va="bottom",
                  fontsize=9, fontweight="bold")
    # the rule separates the predictions on the left from the measurements
    ax_b.axvline(4.5, color="0.35", lw=1.4, ls=(0, (5, 3)))

    ax_b.set_xticks(xb)
    # Rotated to vertical. At 45 degrees the diagonals of neighbouring labels
    # still run into one another at this tick spacing, whatever the font size;
    # upright labels need only their own height between ticks.
    ax_b.set_xticklabels(labels, rotation=90, ha="center", va="top", fontsize=8.5)
    ax_b.set_ylabel(fs.AXIS["E"])
    ax_b.set_ylim(0, 44)
    ax_b.minorticks_on()
    ax_b.tick_params(axis="x", which="minor", bottom=False, top=False)

    # the two panels share a top edge, so the letters land on one line
    fs.panel_letters(fig, [ax_a, ax_b], where="above")

    report = fs.layout_report(fig)
    for key, hits in report.items():
        if hits:
            print(f"  layout: {key}: {hits}")
    fig.savefig(Path(outdir) / "Fig08_rule_of_mixtures.pdf")
    fig.savefig(Path(outdir) / "Fig08_rule_of_mixtures.png", dpi=400)
    plt.close(fig)
    print(f"  wrote Figure_8, {W:.2f} x {H:.2f} in, embed at exactly {W:.2f} in")

    print(zt.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"  laminate: Voigt {v_lam:.2f}  Hill {h_lam:.2f}  Reuss {r_lam:.2f} GPa")
    for cfg in ("basalt_face", "flax_face"):
        m = meas.loc[cfg, "mean"]
        print(f"  {cfg}: {m:.2f} GPa, {100 * (v_lam - m) / v_lam:.1f} % below Voigt")


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
