"""Schematic: where the measurements sit, and in what geometry.

A schematic showing where both mapped windows sit within the full stack, and
the geometry that relates the indentation to the bending test. It draws the
specimen, the polished face and the footprints, all to scale, rather than
comparing window sizes alone, and takes the same care over what is not known.

  (a) The bending specimen in elevation. The polished face is the plane of the
      page: it contains the thickness and the span axis, so the indentation
      axis is the remaining in-plane direction, out of the page. This is the
      relation that makes a modulus measured on this face the right input for a
      beam loaded along the span, and it is drawn rather than asserted.

  (b) The face itself, at true scale against the sixteen plies, with the three
      footprints on it. Their sizes are measured, from the acquired coordinates
      and the mosaic dimensions. Their positions through the thickness are not:
      the absolute placement was never recorded, so they are drawn centred and
      the caption says the placement is indicative.

  (c) Why the missing position matters, stated rather than glossed. Bending
      stress varies linearly through the thickness, so the same window means
      something different at the surface and at the mid-plane. What supports
      the construction is not a known position but a compositional check: the
      fractions measured in the imaged window agree with a calculation from
      fabric areal mass for the whole panel to within 2.2 percentage points.

Everything drawn here is either a specimen dimension from the Methods or a
coordinate range from the deposited tables.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import figstyle as fs

DATA = Path(__file__).resolve().parent.parent / "data"

# Specimen and test geometry, from the Methods.
THICKNESS_MM = 4.60
LENGTH_MM = 80.0
SPAN_MM = 16.0 * THICKNESS_MM          # span-to-thickness ratio of 16:1
N_PLIES = 16
X_RESIN = 380.0                        # end of the embedding resin, in µm

# Mosaic footprint, from the imaged area reported with the segmentation.
MOSAIC_W_MM, MOSAIC_H_MM = 4.39, 1.601


def windows():
    """Footprint of each measurement on the polished face, in millimetres.

    The long axis of both maps runs through the thickness, so the through-
    thickness extent is the larger coordinate range in each case and the
    in-plane extent is the smaller one.
    """
    shallow = pd.read_csv(DATA / "shallow_map.csv")
    kept = shallow[shallow.x_um >= X_RESIN]
    depth = pd.read_csv(DATA / "classified_depth_resolved.csv")

    def extent(df):
        a = (df.x_um.max() - df.x_um.min()) / 1000
        b = (df.y_um.max() - df.y_um.min()) / 1000
        return (max(a, b), min(a, b))          # through-thickness, in-plane

    s_t, s_p = extent(kept)
    d_t, d_p = extent(depth)
    return [("Backscattered-electron mosaic", MOSAIC_H_MM, MOSAIC_W_MM,
             fs.PAPER["basalt"]),
            ("Load-controlled map", s_t, s_p, fs.PAPER["mixed-volume"]),
            ("Depth-resolved map", d_t, d_p, fs.PAPER["matrix"])]


def build(outdir):
    wins = windows()
    ply = THICKNESS_MM / N_PLIES

    fs.apply()
    fig = plt.figure(figsize=(fs.DOUBLE, 0.62 * fs.DOUBLE), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[2.6, 1.0],
                          height_ratios=[1.0, 1.9])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1], sharey=ax_b)

    # ---------------------------------------------------------------- (a)
    ax_a.add_patch(Rectangle((0, 0), LENGTH_MM, THICKNESS_MM,
                             facecolor="0.90", edgecolor="k", linewidth=1.1))
    for k in range(1, N_PLIES):
        ax_a.plot([0, LENGTH_MM], [k * ply] * 2, color="0.72", lw=0.4)
    mid = LENGTH_MM / 2
    for x in (mid - SPAN_MM / 2, mid + SPAN_MM / 2):
        ax_a.plot([x], [-0.35], marker="^", ms=8, color="k", clip_on=False)
    ax_a.annotate("", xy=(mid, THICKNESS_MM + 0.05), xytext=(mid, THICKNESS_MM + 1.9),
                  arrowprops=dict(arrowstyle="-|>", lw=1.6, color="k",
                                  mutation_scale=14))
    ax_a.annotate("", xy=(mid + SPAN_MM / 2, -1.55), xytext=(mid - SPAN_MM / 2, -1.55),
                  arrowprops=dict(arrowstyle="<->", lw=1.1))
    ax_a.text(mid, -1.75, f"support span {SPAN_MM:.1f} mm (16:1)",
              ha="center", va="top", fontsize=8.2)
    ax_a.text(mid, THICKNESS_MM + 1.7, "load", ha="center", va="bottom", fontsize=8.2)

    # The indentation axis leaves the page. Drawn as the usual dotted circle so
    # that the relation between the two experiments is visible at a glance.
    cx, cy = 12.0, THICKNESS_MM / 2
    ax_a.plot([cx], [cy], marker="o", ms=11, mfc="white", mec="k", mew=1.3)
    ax_a.plot([cx], [cy], marker="o", ms=2.6, color="k")
    ax_a.text(cx + 2.6, cy, "indentation axis,\nout of the page", fontsize=8.2,
              va="center", ha="left")
    ax_a.text(LENGTH_MM - 1.0, THICKNESS_MM + 0.35,
              "polished face = plane of the page", fontsize=8.2,
              ha="right", va="bottom")
    ax_a.set_xlim(-3, LENGTH_MM + 3)
    ax_a.set_ylim(-2.6, THICKNESS_MM + 2.6)
    ax_a.set_aspect("equal")
    ax_a.axis("off")

    # ---------------------------------------------------------------- (b)
    view_w = 5.2
    for k in range(N_PLIES):
        ax_b.add_patch(Rectangle((0, k * ply), view_w, ply,
                                 facecolor="0.94" if k % 2 else "0.87",
                                 edgecolor="0.55", linewidth=0.6))
    # Labels live in the margin to the right of the section, not on top of it.
    label_x = view_w + 0.35
    label_y = [4.20, 2.45, 0.72]
    for (name, t_ext, p_ext, colour), ly in zip(wins, label_y):
        x0 = 0.5 * (view_w - p_ext)
        y0 = 0.5 * (THICKNESS_MM - t_ext)
        ax_b.add_patch(Rectangle((x0, y0), p_ext, t_ext, facecolor=colour,
                                 edgecolor="k", linewidth=1.1, alpha=0.92))
        # Leader and label are drawn separately rather than as one annotation.
        # An annotation's bounding box includes its arrow, so the layout check
        # measured each label as though it stretched back to the footprint it
        # points at and reported three collisions that are not on the page.
        ax_b.plot([x0 + p_ext + 0.04, label_x - 0.08],
                  [y0 + t_ext * 0.5, ly], color="0.35", lw=0.8,
                  solid_capstyle="butt", zorder=2)
        ax_b.text(label_x, ly,
                  f"{name}\n{t_ext:.2f} mm through thickness "
                  f"({t_ext / ply:.1f} plies)\n{p_ext:.2f} mm in plane",
                  fontsize=7.0, va="center", ha="left")
    ax_b.set_xlim(0, view_w * 2.05)
    ax_b.set_ylim(0, THICKNESS_MM)
    ax_b.set_xlabel("In-plane, along the span (mm)")
    ax_b.set_ylabel("Through-thickness position (mm)")
    ax_b.set_xticks(np.arange(0, 5.1, 1.0))
    ax_b.set_yticks(np.arange(0, THICKNESS_MM + 0.01, 1.0))
    ax_b.minorticks_on()

    # ---------------------------------------------------------------- (c)
    z = np.linspace(0, THICKNESS_MM, 200)
    ax_c.plot((z - THICKNESS_MM / 2) / (THICKNESS_MM / 2), z,
              color=fs.WONG["red"], lw=1.8)
    ax_c.axvline(0, color="0.6", lw=0.8, ls=(0, (4, 3)))
    fixed = wins[1][1]
    for y0, tag in ((THICKNESS_MM - fixed, "at the surface"),
                    (0.5 * (THICKNESS_MM - fixed), "at mid-plane")):
        ax_c.add_patch(Rectangle((-1.05, y0), 2.10, fixed, facecolor="0.5",
                                 alpha=0.22, edgecolor="none"))
        ax_c.text(-0.98, y0 + fixed / 2, f"0.70 mm window\n{tag}", fontsize=7.0,
                  va="center", ha="left", color="0.25")
    ax_c.set_xlim(-1.15, 1.15)
    ax_c.set_xlabel("Bending stress,\n$\\sigma/\\sigma_{max}$")
    ax_c.set_xticks([-1, 0, 1])
    ax_c.tick_params(labelleft=False)
    ax_c.minorticks_on()

    # Set back by hand: the tall y-axis label on (b) leaves no room for the
    # measured set-back that panel_letters would use, and the letter lands on it.
    fs.panel_label(ax_a, "a", dx=-0.055, dy=1.30)
    fs.panel_label(ax_b, "b", dx=-0.205, dy=1.045)
    fs.panel_label(ax_c, "c", dx=-0.230, dy=1.045)
    fs.save(fig, "Reply_R4_map_coverage", outdir)
    for name, t_ext, p_ext, _ in wins:
        print(f"  {name:34s} {t_ext:.3f} x {p_ext:.3f} mm  "
              f"= {t_ext / ply:.2f} plies through thickness")


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
