"""Figure 6 - the mechanical classes in property space at z = 150 nm.

Rebuilt from the deposited per-indent table, matching the submitted figure
element for element:

  (a) elastic modulus against hardness, one point per indent, coloured by class,
      with the QC/invalid population shown in grey so the reader can see what
      the filter removed. The two dotted horizontal lines are the modulus
      thresholds that define the rule, 15 and 40 GPa; they are horizontal
      because modulus is on the vertical axis.
  (b, c, d) per-class boxplots of modulus, hardness and phase angle at the same
      depth, on the same axis limits as (a) where the quantity is shared.

Axes are placed explicitly. Constrained layout packs the canvas edge to edge,
which leaves no room outside a panel for its letter.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

DATA = Path(__file__).resolve().parent.parent / "data"
ORDER = ["flax", "matrix", "mixed-volume", "basalt"]
LABEL = {"flax": "flax", "matrix": "matrix",
         "mixed-volume": "basalt-mixed", "basalt": "basalt"}
E_LOWER, E_UPPER = 15.0, 40.0          # the modulus thresholds of the rule
LIMS = {"E_150": (0, 100), "H_150": (0, 8), "PA_150": (0, 15)}


def build(outdir):
    # Created here rather than assumed. Run from a fresh checkout with an
    # output directory that does not exist yet, these scripts used to fail
    # on the savefig at the end, after doing all the work.
    Path(outdir).mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(DATA / "classified_depth_resolved.csv")
    named = lab[lab.constituent.isin(ORDER)]

    # four panels across a 6.3 in row is tight: the bold preset keeps its heavy
    # boxed axes and palette, but the type is scaled to what actually fits
    fs.set_style("bold", font_scale=0.55)

    # Two-pass layout. Each panel hangs a different distance to the left of its
    # axes box, because the label, the tick labels and the letter differ in
    # width from panel to panel. A single uniform gap has to be as wide as the
    # worst case, which wastes width on the other three. The figure is drawn
    # once to measure the real overhangs, then rebuilt with the gaps those
    # measurements imply, which is worth about 13 % of panel width here.
    W = 6.60
    RIGHT, TOP = 0.04, 0.30
    PANEL_H, BOTTOM = 1.58, 0.86
    H = TOP + PANEL_H + BOTTOM
    PAD = 0.03                       # breathing space between panels

    def layout(gaps):
        fig = plt.figure(figsize=(W, H))
        panel_w = (W - sum(gaps) - RIGHT) / 4
        axes, x = [], 0.0
        for i in range(4):
            x += gaps[i]
            axes.append(fig.add_axes([x / W, BOTTOM / H, panel_w / W, PANEL_H / H]))
            x += panel_w
        return fig, axes, panel_w

    def overhangs(fig, axes):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        return [(ax.get_window_extent(r).x0 - ax.get_tightbbox(r).x0) / fig.dpi
                for ax in axes]


    def draw(axes):
        ax_a, ax_b, ax_c, ax_d = axes
        _draw_panels(lab, ax_a, ax_b, ax_c, ax_d)
        # above the panel rather than beside it: much closer, since the letter no
        # longer has to clear the rotated axis label
        fs.panel_letters(fig, list(axes), fontsize=11, where="above")

    # first pass: provisional gaps, only to measure what each panel really needs
    fig, axes, _ = layout([0.60] * 4)
    draw(axes)
    need = overhangs(fig, axes)
    plt.close(fig)

    # second pass: the gaps the measurement asks for
    fig, axes, panel_w = layout([n + PAD for n in need])
    draw(axes)
    ax_a, ax_b, ax_c, ax_d = axes

    report = fs.layout_report(fig)
    for key, hits in report.items():
        if hits:
            print(f"  layout: {key}: {hits}")

    fig.savefig(Path(outdir) / "Fig06_property_relationships.pdf")
    fig.savefig(Path(outdir) / "Fig06_property_relationships.png", dpi=400)
    plt.close(fig)
    print(f"  wrote Figure_6, {W:.2f} x {H:.2f} in, embed at exactly {W:.2f} in")
    print(f"  panel width {panel_w:.3f} in   measured overhangs "
          f"{[round(n, 3) for n in need]}")
    for cls in ORDER + ["QC/invalid"]:
        print(f"    {cls:13s} n = {(lab.constituent == cls).sum()}")


def _draw_panels(lab, ax_a, ax_b, ax_c, ax_d):
    named = lab[lab.constituent.isin(ORDER)]
    # ---- (a) modulus against hardness, every indent -----------------------
    for cls in ORDER + ["QC/invalid"]:
        sub = lab[lab.constituent == cls]
        colour = fs.PAPER["qc"] if cls == "QC/invalid" else fs.CONSTITUENT[cls]
        name = "QC/invalid" if cls == "QC/invalid" else LABEL[cls]
        ax_a.scatter(sub.H_150, sub.E_150, s=3.5, alpha=0.75, linewidths=0,
                     color=colour, label=f"{name}  (n={len(sub)})")
    for t in (E_LOWER, E_UPPER):
        ax_a.axhline(t, color="0.25", lw=0.8, ls=(0, (1.6, 1.6)), zorder=1)
    ax_a.set_xlabel(fs.AXIS["H"])
    ax_a.set_ylabel(fs.AXIS["E"])
    ax_a.set_xlim(*LIMS["H_150"])
    ax_a.set_ylim(*LIMS["E_150"])
    ax_a.set_xticks([0, 2, 4, 6, 8])          # as submitted; the default
    ax_a.set_yticks([0, 20, 40, 60, 80, 100])  # locator gives only 0 and 5
    leg = ax_a.legend(loc="upper left", fontsize=5.2, frameon=False,
                      handletextpad=0.25, borderpad=0.15, labelspacing=0.28,
                      markerscale=2.6)
    for h in leg.legend_handles:
        h.set_alpha(1.0)

    # ---- (b, c, d) per-class distributions --------------------------------
    for ax, col, ylab in ((ax_b, "E_150", fs.AXIS["E"]),
                          (ax_c, "H_150", fs.AXIS["H"]),
                          (ax_d, "PA_150", fs.AXIS["PA"])):
        data = [named.loc[named.constituent == c, col].dropna().values for c in ORDER]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color="k", lw=1.3),
                        whiskerprops=dict(lw=0.9), capprops=dict(lw=0.9),
                        boxprops=dict(lw=0.9))
        for patch, cls in zip(bp["boxes"], ORDER):
            patch.set_facecolor(fs.CONSTITUENT[cls])
            patch.set_edgecolor("k")
        ax.set_xticks(range(1, len(ORDER) + 1))
        # upright: at 30 degrees the diagonals of four labels under a panel this
        # narrow run into one another
        ax.set_xticklabels([LABEL[c] for c in ORDER], rotation=90,
                           ha="center", va="top")
        ax.set_ylabel(ylab)
        ax.set_ylim(*LIMS[col])

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.minorticks_on()
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
