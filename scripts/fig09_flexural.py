"""Figure 9 - three-point bending response, all nine specimens.

Reviewer 3 comment 22 asked for every replicate instead of one specimen, and for
the removal of the +/- 1 sigma envelope that had been drawn around a single
curve. Both are done here. Every curve the deposit holds is drawn: five
basalt-longitudinal and four flax-longitudinal, the flax configuration being four
because the source workbook held one of its curves twice and the duplicate is not
deposited.

Panel (a) is the stress-strain family with the ASTM D790 chord window shaded.
Panel (b) is the tangent modulus of the same specimens, which is where the two
configurations actually differ: they cross at about 0.28 % strain and the
flax-longitudinal laminate softens first.

Each panel carries a narrow strip on its right, sharing its vertical axis, that
reduces the family to the number quoted in the text: the flexural strength in
(a) and the ASTM D790 chord modulus in (b), each as a mean over the specimens of
that configuration with its standard deviation. Sharing the axis is the point, so
the summary sits
level with the curves it comes from and can be read against them directly.

Strain is toe-compensated per ASTM D790 Annex A1.1 in both panels, using the
intercepts in flexural_moduli.csv, so the curves start at the origin and the
chord window means the same on the plot as in the calculation.

Axes are placed at explicit sizes rather than by constrained layout. The layout
engine packs the canvas edge to edge, which leaves no room for a panel letter
outside the axes: the letter then lands on the tick labels or on the axis label.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

DATA = Path(__file__).resolve().parent.parent / "data"
WINDOW = (0.05, 0.25)          # ASTM D790 chord window, per cent strain
SMOOTH_PCT = 0.09              # Savitzky-Golay window for the derivative


def tangent_modulus(strain_pct, stress_mpa):
    """Local slope of the stress-strain curve, in GPa.

    Savitzky-Golay differentiation on a uniform strain grid. The window spans
    about 0.09 % strain: shorter and the digitiser step shows as ripple on the
    derivative, longer and it starts to blur the 0.20 % over which the two
    configurations differ.
    """
    grid = np.linspace(strain_pct.min(), strain_pct.max(), 1200)
    stress = np.interp(grid, strain_pct, stress_mpa)
    step = grid[1] - grid[0]
    win = max(41, int(round(SMOOTH_PCT / step)) | 1)
    dsig = savgol_filter(stress, win, 3, deriv=1, delta=step)
    # stress is MPa and strain is per cent, so MPa per %-strain -> GPa is x0.1
    return grid, dsig * 0.1


def find_crossover(per_config, lo=0.15, hi=0.55):
    """Strain at which the two configurations' mean tangent moduli cross."""
    grid = np.linspace(lo, hi, 2000)
    a, b = (np.vstack([np.interp(grid, g, t) for g, t in per_config[c]]).mean(axis=0)
            for c in ("basalt_face", "flax_face"))
    d = a - b
    sign = np.where(np.diff(np.signbit(d)))[0]
    if len(sign) == 0:
        return None
    i = sign[0]
    return grid[i] - d[i] * (grid[i + 1] - grid[i]) / (d[i + 1] - d[i])


def build(outdir):
    # Created here rather than assumed. Run from a fresh checkout with an
    # output directory that does not exist yet, these scripts used to fail
    # on the savefig at the end, after doing all the work.
    Path(outdir).mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(DATA / "flexural_curves.csv")
    moduli = pd.read_csv(DATA / "flexural_moduli.csv").set_index("specimen_id")

    fs.set_style("bold", font_scale=0.78)

    W = 6.60
    LEFT, GAP, RIGHT, TOP = 0.86, 0.62, 0.08, 0.34
    STRIP_W, STRIP_GAP = 0.62, 0.05
    PANEL_H, BOTTOM = 2.30, 0.75
    MAIN_W = (W - LEFT - 2 * (STRIP_W + STRIP_GAP) - GAP - RIGHT) / 2
    H = TOP + PANEL_H + BOTTOM
    fig = plt.figure(figsize=(W, H))

    def place(x_in, w_in):
        return fig.add_axes([x_in / W, BOTTOM / H, w_in / W, PANEL_H / H])

    ax_a = place(LEFT, MAIN_W)
    sa = place(LEFT + MAIN_W + STRIP_GAP, STRIP_W)
    x0b = LEFT + MAIN_W + STRIP_GAP + STRIP_W + GAP
    ax_b = place(x0b, MAIN_W)
    sb = place(x0b + MAIN_W + STRIP_GAP, STRIP_W)

    for ax in (ax_a, ax_b):
        ax.axvspan(*WINDOW, color="0.85", lw=0, zorder=0)

    seen, per_config = set(), {"basalt_face": [], "flax_face": []}
    for spec, g in curves.groupby("specimen_id", sort=True):
        cfg = g["configuration"].iloc[0]
        colour = fs.CONFIG[cfg]
        e = g["strain_pct"].to_numpy() - moduli.loc[spec, "toe_intercept_pct"]
        s = g["stress_MPa"].to_numpy()
        keep = e >= 0
        e, s = e[keep], s[keep]

        label = fs.CONFIG_LABEL[cfg] if cfg not in seen else None
        seen.add(cfg)
        ax_a.plot(e, s, color=colour, lw=1.3, alpha=0.9, label=label)

        grid, tan = tangent_modulus(e, s)
        per_config[cfg].append((grid, tan))
        show = grid <= 0.60      # beyond this the response is damage, not stiffness
        ax_b.plot(grid[show], tan[show], color=colour, lw=1.3, alpha=0.9)

    crossover = find_crossover(per_config)

    ax_a.set_xlabel("Flexural strain (%)")
    ax_a.set_ylabel("Flexural stress (MPa)")
    ax_a.set_xlim(0, 1.45)
    ax_a.set_ylim(0, 200)
    ax_a.legend(loc="lower right", frameon=False, handlelength=1.5, fontsize=9)

    ax_b.set_xlabel("Flexural strain (%)")
    ax_b.set_ylabel("Tangent modulus (GPa)")
    ax_b.set_xlim(0, 0.60)
    ax_b.set_ylim(8, 24)
    ax_b.axvline(crossover, color="k", lw=1.0, ls=(0, (4, 3)))
    # Two lines to the right of the rule. One line, either side, is about
    # 1.0 in wide against a 1.7 in panel, so it always straddles a spine: set
    # left it crosses the y axis, set right it crosses the right spine. Two
    # short lines fit. assert_inside below measures this rather than assuming it.
    cross_label = ax_b.text(crossover + 0.012, 23.4,
                            f"crossover\n{crossover:.2f} %",
                            fontsize=8.5, ha="left", va="top", linespacing=1.15)

    for ax in (ax_a, ax_b):
        ax.minorticks_on()

    # ---- summary strips, sharing the vertical axis of the panel beside them --
    strength = (curves.groupby(["configuration", "specimen_id"]).stress_MPa.max()
                .groupby("configuration").agg(["mean", "std"]))
    chord = (moduli.reset_index()
             .groupby("configuration")["E_D790_toe_compensated_GPa"].agg(["mean", "std"]))

    # Each strip is printed at the precision its own caption sentence uses. The
    # caption gives the strength to one decimal and the chord modulus to two, and
    # (b) used to print one for both, so the panel read 18.8 +/- 0.5 beside a
    # caption saying 18.85 +/- 0.54. Figure 8 carries the same moduli and made the
    # same move for the same reason. Two decimals cost two glyphs, which the
    # containment check below measures against the strip width rather than
    # assuming; at 7 pt they fit with room to spare in a 0.62 in strip.
    strip_labels = []
    for strip, table, main, unit, places in ((sa, strength, ax_a, "MPa", 1),
                                             (sb, chord, ax_b, "GPa", 2)):
        lo, hi = main.get_ylim()
        span = hi - lo
        means = {c: table.loc[c, "mean"] for c in ("basalt_face", "flax_face")}
        close = abs(means["basalt_face"] - means["flax_face"]) < 0.12 * span
        for i, cfg in enumerate(("basalt_face", "flax_face")):
            m, sd = means[cfg], table.loc[cfg, "std"]
            strip.errorbar(i, m, yerr=sd, fmt="o", ms=7, capsize=5,
                           color=fs.CONFIG[cfg], mec="k", mew=1.0, lw=1.6, zorder=3)
            # a label goes above its point unless there is no headroom there, or
            # unless the two means are close enough that the labels would collide
            no_room = (m + sd) > hi - 0.14 * span
            below = no_room or (close and cfg == "basalt_face")
            # centred on the strip, not on the point: a label centred on a
            # point sits half outside a strip this narrow and is clipped away.
            # Colour ties each value to its marker.
            strip_labels.append(strip.annotate(f"{m:.{places}f} ± "
                           f"{sd:.{places}f}", xy=(0.5, m),
                           xytext=(0, -17 if below else 11),
                           textcoords="offset points", ha="center",
                           va="top" if below else "bottom", fontsize=7,
                           color=fs.CONFIG[cfg], fontweight="bold"))
        strip.set_ylim(lo, hi)
        strip.set_xlim(-0.5, 1.5)
        strip.set_yticklabels([])
        strip.set_xticks([0, 1])
        strip.set_xticklabels(["bas.", "flax"], fontsize=7.5)
        strip.tick_params(axis="y", which="both", length=0)
        strip.set_xlabel(unit, fontsize=8.5)

    # the two panels share a top edge, so the letters land on one line
    fs.panel_letters(fig, [ax_a, ax_b], where="above")

    print("  label containment:")
    ok = fs.assert_inside(ax_b, cross_label, name="panel b")
    ok &= fs.assert_inside(sa, strip_labels[:2], name="strip a")
    ok &= fs.assert_inside(sb, strip_labels[2:], name="strip b")
    if not ok:
        raise SystemExit("a label overflows its axes; fix before shipping")

    report = fs.layout_report(fig)
    for key, hits in report.items():
        if hits:
            print(f"  layout: {key}: {hits}")

    fig.savefig(Path(outdir) / "Fig09_three_point_bending.pdf")
    fig.savefig(Path(outdir) / "Fig09_three_point_bending.png", dpi=400)
    plt.close(fig)
    print(f"  wrote Figure_9, {W:.2f} x {H:.2f} in, embed at exactly {W:.2f} in")
    print(f"  strength  {strength.to_dict()}")
    print(f"  chord     {chord.to_dict()}")
    print(f"  measured crossover: {crossover:.4f} % strain")
    for cfg, g in moduli.groupby("configuration"):
        v = g["E_D790_toe_compensated_GPa"]
        print(f"  {cfg}: {v.mean():.2f} +/- {v.std(ddof=1):.2f} GPa (n={len(v)})")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
