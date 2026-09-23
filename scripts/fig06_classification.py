"""Figure 5 - per-indent constituent classification of the two acquisitions.

Each classification map is drawn at the same physical size, on the same axis
ranges and in the same orientation as the corresponding property map, so panel
(a) can be laid over the Figure 3 maps and panel (b) over the Figure 4 strips
and they coincide.

The sizes were measured from the submitted figures rather than assumed. In
Figure 3 each map is 5.103 x 1.470 in on a 7.20 in page, embedded in the
manuscript at 6.00 in, so it prints at 4.253 x 1.225 in. In Figure 4 each strip
is 5.370 x 1.075 in on an 8.27 in page, embedded at 5.50 in, so it prints at
3.571 x 0.715 in. Note that Figure 3 is not drawn at the data aspect: the map
spans 703 x 97 um, an aspect of 7.2, but is drawn at 3.47, so it is stretched
vertically by a factor of two. Equal-aspect axes could never match it, and the
axes here are therefore placed at explicit sizes with aspect="auto".

Panel (c) reports the indent-class fractions. They are fractions of classified
indents, not constituent volume fractions: inside the flax-rich half the rule
assigns every compliant indent to flax, so matrix there is counted as flax. The
measured constituent volume fractions are in Figure 2.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

DATA = Path(__file__).resolve().parent.parent / "data"
CLASSES = [("flax", fs.PAPER["flax"]), ("matrix", fs.PAPER["matrix"]),
           ("mixed-volume", fs.PAPER["mixed-volume"]), ("basalt", fs.PAPER["basalt"])]
QC = fs.PAPER["qc"]
CODE = {n: i + 1 for i, (n, _) in enumerate(CLASSES)}


def grid(df, xcol, ycol):
    """Class index on the acquisition lattice, with the extent in micrometres."""
    xs, ys = np.sort(df[xcol].unique()), np.sort(df[ycol].unique())
    out = np.full((len(ys), len(xs)), np.nan)
    out[np.searchsorted(ys, df[ycol].values),
        np.searchsorted(xs, df[xcol].values)] = df.constituent.map(CODE).values
    return out, (xs.min(), xs.max(), ys.min(), ys.max())


def draw(ax, arr, extent):
    ax.imshow(np.zeros(arr.shape), origin="lower", extent=extent, aspect="auto",
              cmap=ListedColormap([QC]), interpolation="nearest")
    ax.imshow(arr, origin="lower", extent=extent, aspect="auto",
              cmap=ListedColormap([c for _, c in CLASSES]),
              norm=BoundaryNorm(np.arange(0.5, len(CLASSES) + 1.5), len(CLASSES)),
              interpolation="nearest")


def build(outdir):
    # Created here rather than assumed. Run from a fresh checkout with an
    # output directory that does not exist yet, these scripts used to fail
    # on the savefig at the end, after doing all the work.
    Path(outdir).mkdir(parents=True, exist_ok=True)
    shallow = pd.read_csv(DATA / "classified_shallow.csv")
    depth = pd.read_csv(DATA / "classified_depth_resolved.csv")

    # (a) as Figure 3: x horizontal, y vertical
    a_arr, a_ext = grid(shallow, "x_um", "y_um")
    # (b) as Figure 4: the long axis, y, horizontal
    b_arr, b_ext = grid(depth, "y_um", "x_um")

    fs.set_style("bold", font_scale=0.78)

    # Printed size of the corresponding maps, in inches, measured from the
    # submitted Figures 3 and 4 (see the module docstring).
    A_W, A_H = 4.253, 1.225          # a Figure 3 map, as it prints
    B_W, B_H = 3.571, 0.715          # a Figure 4 strip, as it prints

    # Every panel is drawn to one common width so the figure reads as a set.
    # Panel (a) keeps the Figure 3 width, so it still registers with Figure 3
    # one to one. Panel (b) is scaled up to match, preserving its own aspect,
    # which means it is a factor of A_W / B_W larger than the Figure 4 strip
    # rather than identical to it. That is a deliberate trade: internal
    # consistency of this figure over pixel registration with Figure 4.
    PANEL_W = A_W
    SCALE_B = PANEL_W / B_W
    PANEL_A = (PANEL_W, A_H)
    PANEL_B = (PANEL_W, B_H * SCALE_B)

    LEFT, RIGHT, TOP = 0.78, 0.26, 0.34
    GAP_LABEL = 0.62          # room under a map for its x label and ticks
    LEG_H, BAR_H, BOTTOM = 0.34, 2.05, 0.78
    W = LEFT + PANEL_W + RIGHT
    H = (TOP + PANEL_A[1] + GAP_LABEL + PANEL_B[1] + GAP_LABEL
         + LEG_H + BAR_H + BOTTOM)

    fig = plt.figure(figsize=(W, H))

    def place(x_in, y_top_in, w_in, h_in):
        """Axes at an exact size in inches, measured from the top of the canvas."""
        return fig.add_axes([x_in / W, 1 - (y_top_in + h_in) / H, w_in / W, h_in / H])

    y = TOP
    ax_a = place(LEFT, y, *PANEL_A)
    draw(ax_a, a_arr, a_ext)
    ax_a.set_xlabel("x (µm)"); ax_a.set_ylabel("y (µm)")

    y += PANEL_A[1] + GAP_LABEL
    ax_b = place(LEFT, y, *PANEL_B)
    draw(ax_b, b_arr, b_ext)
    ax_b.set_xlabel("y (µm)"); ax_b.set_ylabel("x (µm)")

    y += PANEL_B[1] + GAP_LABEL
    ax_leg = place(LEFT, y, PANEL_W, LEG_H)
    ax_leg.axis("off")
    ax_leg.legend(handles=[Patch(facecolor=c, edgecolor="k", lw=0.8, label=n)
                           for n, c in CLASSES]
                          + [Patch(facecolor=QC, edgecolor="k", lw=0.8, label="QC / invalid")],
                  loc="center", ncol=5, frameon=False, handlelength=1.2,
                  columnspacing=1.6)

    y += LEG_H + 0.22        # clearance for the letter above the bar panel
    ax_c = place(LEFT, y, PANEL_W, BAR_H)
    names = [n for n, _ in CLASSES]
    x = np.arange(len(names))
    for k, (df, label, hatch) in enumerate([(shallow, "load-controlled survey", ""),
                                            (depth, "depth-resolved survey", "//")]):
        d = df[df.constituent.isin(names)]
        frac = [100 * (d.constituent == n).mean() for n in names]
        ax_c.bar(x + (k - 0.5) * 0.38, frac, 0.36,
                 color=[c for _, c in CLASSES], edgecolor="k", linewidth=1.2,
                 hatch=hatch, label=label, zorder=3)
        for xi, f in zip(x + (k - 0.5) * 0.38, frac):
            ax_c.text(xi, f + 1.4, f"{f:.1f}", ha="center", va="bottom", fontsize=9)
    ax_c.set_xticks(x); ax_c.set_xticklabels(names)
    ax_c.set_ylabel("Indent-class fraction (%)")
    ax_c.set_ylim(0, 84)
    ax_c.legend(loc="upper right", frameon=False, handlelength=1.3)
    ax_c.minorticks_on()
    ax_c.tick_params(axis="x", which="minor", bottom=False, top=False)

    # all three panels share a left edge, so the letters fall on one vertical line
    fs.panel_letters(fig, [ax_a, ax_b, ax_c], where="above")

    report = fs.layout_report(fig)
    for key, hits in report.items():
        if hits:
            print(f"  layout: {key}: {hits}")
    fig.savefig(str(Path(outdir) / "Fig06_constituent_classification.pdf"))
    fig.savefig(str(Path(outdir) / "Fig06_constituent_classification.png"), dpi=400)
    plt.close(fig)
    print("  wrote Fig06_constituent_classification.pdf and .png")
    print(f"  figure {W:.3f} x {H:.3f} in, embed at exactly {W:.3f} in")
    print(f"  panel a {PANEL_A[0]:.3f} x {PANEL_A[1]:.3f} in   (Figure 3 map, 1:1)")
    print(f"  panel b {PANEL_B[0]:.3f} x {PANEL_B[1]:.3f} in   "
          f"(Figure 4 strip scaled by {SCALE_B:.3f})")
    print(f"  panel c {PANEL_W:.3f} in wide")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
