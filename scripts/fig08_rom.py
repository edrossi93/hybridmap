"""Figure 8 - the rule-of-mixtures construction, within zone and at the laminate scale.

Panel (a) works inside the depth-resolved strip, which is banded into three
architectural zones (flax-rich bottom, basalt-rich centre, flax-rich top). For
each zone the within-zone class fractions and the class median moduli at
z = 150 nm give Voigt, Hill and Reuss. A fourth bar shows the shallow-corrected
Hill bound, in which the basalt input is replaced by the 86 GPa median measured
in the shallow map, where the fibres are not depressed by substrate compliance.
The zone-averaged modulus actually measured is marked so the reader can see
which bound lands closest.

Panel (b) is the laminate-scale construction from the shallow map. It has two
levels: the class fractions and medians inside each zone come from the indents,
and the share the two zones occupy comes from the imaging of the same section.
A second Voigt bar shows what the indent counts give instead, which describes
the window that was mapped rather than the section; the difference between the
two bars is the zone-share term of the uncertainty budget. The two measured
flexural moduli are drawn beside them with their standard deviations.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs
import zone_share as zs
import areal_mass_fractions
from rule_of_mixtures import bounds, class_table, CLASSES

DATA = Path(__file__).resolve().parent.parent / "data"

# median modulus of the pure-axial basalt class in the shallow acquisition
BASALT_SHALLOW_GPA = 86.0

# Tabulated-input baselines quoted in §3.4, using tabulated constituents and the
# measured fibre volume fraction of 0.649.
TABULATED_VOIGT = 35.75

# The tabulated constituent inputs of §3.4. Basalt and flax are the upper
# midpoints of the ranges cited in the Introduction, the matrix is the bulk
# PA-11 value, and the fibre volume fraction is the one calculated from the
# fabric areal mass and the constituent densities.
HT_E_BASALT = 80.0
HT_E_FLAX = 60.0
HT_E_MATRIX = 1.5
# Derived rather than written down: `areal_mass_fractions.py` owns that route and
# names its inputs, and this was a literal with nothing behind it. Rounded to the
# three decimals §3.4 prints, which is the number the panel was drawn from, so the
# bar does not move by a hundredth of a per cent of fibre.
HT_V_FIBRE = round(areal_mass_fractions.fibre_volume_fraction(), 3)
# Reinforcement geometry factor for transverse loading of continuous circular
# fibres, which is the conventional value and the only one the paper's inputs
# support: no aspect ratio is measured here, and the paper claims none.
HT_XI = 2.0

# Standard uncertainty on the section-weighted estimate, from
# uncertainty_budget.py. Drawn on the first bar so that the figure carries the
# same interval the text quotes.
BUDGET_SD = 1.44

ZONE_ORDER = ["flax-bottom", "basalt-centre", "flax-top"]
# kept short: three category names under a narrow panel run into one another
ZONE_LABEL = {"flax-bottom": "bottom", "basalt-centre": "centre",
              "flax-top": "top"}


def halpin_tsai_balanced(e_basalt=HT_E_BASALT, e_flax=HT_E_FLAX,
                         e_matrix=HT_E_MATRIX, v_fibre=HT_V_FIBRE, xi=HT_XI):
    """The Halpin-Tsai baseline of §3.4, derived rather than asserted.

    A balanced 0/90 average of a single lamina: the longitudinal direction is
    the iso-strain sum, the transverse direction is the Halpin-Tsai estimate,
    and the average of the two is what a balanced fabric of the same constituents
    would give. The two fibre types enter through one fibre modulus, the mean of
    the two tabulated values, because the tabulated route has no measurement that
    would split them by volume and the fabric specification gives the split by
    mass.

    Returns the longitudinal, transverse and averaged moduli in GPa. Nothing is
    printed here; report_halpin_tsai() below does that, so that the value can be
    taken at import time without output appearing as a side effect.
    """
    e_fibre = 0.5 * (e_basalt + e_flax)
    ratio = e_fibre / e_matrix
    eta = (ratio - 1.0) / (ratio + xi)
    longitudinal = v_fibre * e_fibre + (1.0 - v_fibre) * e_matrix
    transverse = e_matrix * (1.0 + xi * eta * v_fibre) / (1.0 - eta * v_fibre)
    balanced = 0.5 * (longitudinal + transverse)
    return longitudinal, transverse, balanced


# The Halpin-Tsai baseline of the panel, derived from the inputs §3.4 states
# rather than copied out of the text. It used to be the literal 27.1, which was
# the figure the manuscript carried and which no convention applied to those
# inputs reproduces: reaching it needs a fibre volume fraction of 0.646, a fibre
# modulus of 69.6 GPa or a matrix modulus of 1.45 GPa in place of the values the
# paper states. §3.4 now prints 27.2, which is what the line below rounds to, so
# the bar and the sentence come from one calculation.
TABULATED_HT = halpin_tsai_balanced()[2]


def report_halpin_tsai(e_basalt=HT_E_BASALT, e_flax=HT_E_FLAX,
                       e_matrix=HT_E_MATRIX, v_fibre=HT_V_FIBRE, xi=HT_XI):
    """Print the baseline with every input beside it.

    The point of doing this on every run is that the baseline has inputs at all,
    which the literal it replaced did not.
    """
    longitudinal, transverse, balanced = halpin_tsai_balanced(
        e_basalt, e_flax, e_matrix, v_fibre, xi)
    e_fibre = 0.5 * (e_basalt + e_flax)
    print(f"  Halpin-Tsai inputs: basalt {e_basalt:.1f}, flax {e_flax:.1f}, "
          f"fibre mean {e_fibre:.1f}, matrix {e_matrix:.2f} GPa, "
          f"v_fibre {v_fibre:.3f}, xi {xi:.1f}")
    print(f"  longitudinal {longitudinal:.2f}, transverse {transverse:.2f}, "
          f"balanced 0/90 average {balanced:.2f} GPa, which is the "
          f"{balanced:.1f} GPa of §3.4")
    return longitudinal, transverse, balanced


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

    # The laminate construction, weighted two ways. The section weighting is
    # the one the paper reports; the window weighting is kept beside it so the
    # reader can see how much of the figure the placement of the map decides.
    tiles = pd.read_csv(DATA / "bse_segmentation_tiles.csv")
    per_zone, medians, w_window = zs.zone_fractions(shallow)
    w_section = (tiles.n_basaltzone.sum()
                 / (tiles.n_basaltzone.sum() + tiles.n_flaxzone.sum()))
    _, (v_lam, h_lam, r_lam) = zs.at_share(per_zone, medians, w_section)
    _, (v_window, _, _) = zs.at_share(per_zone, medians, w_window)
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
    labels = ["Voigt", "Voigt (window-weighted)", "Hill", "Reuss",
              "Voigt (tabulated inputs)", "Halpin-Tsai (tabulated inputs)",
              "Basalt-long.", "Flax-long."]
    values = [v_lam, v_window, h_lam, r_lam, TABULATED_VOIGT, TABULATED_HT,
              meas.loc["basalt_face", "mean"], meas.loc["flax_face", "mean"]]
    errs = [BUDGET_SD, 0, 0, 0, 0, 0,
            meas.loc["basalt_face", "std"], meas.loc["flax_face", "std"]]
    cols = [fs.PAPER["voigt"], fs.PAPER["voigt"], fs.PAPER["hill"],
            fs.PAPER["reuss"], fs.PAPER["corrected"], fs.PAPER["corrected"],
            fs.PAPER["basalt"], fs.PAPER["flax"]]
    hatch = ["//", "xx", "//", "//", "//", "//", "", ""]

    xb = np.arange(len(labels))
    ax_b.bar(xb, values, 0.66, yerr=errs, capsize=4, color=cols, edgecolor="k",
             linewidth=1.1, zorder=3, error_kw=dict(lw=1.4))
    for patch, h in zip(ax_b.patches, hatch):
        patch.set_hatch(h)
    # Two decimals, which is the precision the caption uses for the estimate and
    # for the two measured moduli: at one decimal the first bar read 18.5 beside
    # a caption saying 18.50 +/- 1.44, and the two measured bars read 18.8 and
    # 19.8 beside 18.85 and 19.76. Two decimals are two glyphs wider, which at
    # 9 pt brings the first pair and the last pair into contact, by about a pixel
    # each, too little for layout_report() to call it a collision and enough to
    # see. 8 pt leaves four pixels between every neighbouring pair and sits just
    # under the tick labels, which are 8.5.
    for xi, val, e in zip(xb, values, errs):
        ax_b.text(xi, val + e + 0.8, f"{val:.2f}", ha="center", va="bottom",
                  fontsize=8, fontweight="bold")
    # the rule separates the predictions on the left from the measurements
    ax_b.axvline(5.5, color="0.35", lw=1.4, ls=(0, (5, 3)))

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

    report_halpin_tsai()

    print(zt.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"  laminate, section-weighted: Voigt {v_lam:.2f}  Hill {h_lam:.2f}"
          f"  Reuss {r_lam:.2f} GPa   (window-weighted Voigt {v_window:.2f})")
    for cfg in ("basalt_face", "flax_face"):
        m = meas.loc[cfg, "mean"]
        print(f"  {cfg}: {m:.2f} GPa, {100 * (m - v_lam) / v_lam:+.1f} % "
              f"relative to the estimate")


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
