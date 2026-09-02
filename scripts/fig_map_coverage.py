"""Reply figure: how much of the laminate each measurement covers.

Reviewer 3 comment 12 asks where the mapped windows sit within the full stack.
What the records support is the extent of each window, taken from the acquired
coordinates, against the sixteen-ply thickness. The absolute through-thickness
position of each window was not recorded at acquisition, so the windows are
drawn at their true size but their placement in the stack is indicative and the
caption says so. Inventing a position would be worse than admitting the gap.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

import figstyle as fs

DATA = Path(__file__).resolve().parent.parent / "data"
THICKNESS_MM = 4.60
N_PLIES = 16
X_RESIN = 380.0


def build(outdir):
    shallow = pd.read_csv(DATA / "shallow_map.csv")
    kept = shallow[shallow.x_um >= X_RESIN]
    depth = pd.read_csv(DATA / "classified_depth_resolved.csv")

    # through-thickness extent of each measurement, in millimetres
    windows = [
        ("Backscattered-electron mosaic", 1.601, fs.PAPER["basalt"]),
        ("Fixed-depth map (after resin cut)",
         (kept.x_um.max() - kept.x_um.min()) / 1000, fs.PAPER["mixed-volume"]),
        ("Depth-resolved map",
         (depth.y_um.max() - depth.y_um.min()) / 1000, fs.PAPER["matrix"]),
    ]

    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.ONE_HALF, 0.85 * fs.ONE_HALF), layout="constrained")

    ply = THICKNESS_MM / N_PLIES
    for k in range(N_PLIES):
        ax.add_patch(Rectangle((0, k * ply), 1.0, ply,
                               facecolor="0.93" if k % 2 else "0.86",
                               edgecolor="k", linewidth=0.7))
    ax.text(0.5, THICKNESS_MM + 0.10, f"{N_PLIES} plies, {THICKNESS_MM:.2f} mm",
            ha="center", va="bottom", fontsize=10)

    for i, (name, extent, colour) in enumerate(windows):
        x = 1.45 + i * 0.75
        y0 = 0.5 * (THICKNESS_MM - extent)
        ax.add_patch(Rectangle((x, y0), 0.45, extent, facecolor=colour,
                               edgecolor="k", linewidth=1.2, alpha=0.9))
        ax.text(x + 0.225, y0 + extent + 0.06,
                f"{extent:.2f} mm\n{extent / ply:.1f} plies",
                ha="center", va="bottom", fontsize=9)
        ax.text(x + 0.225, -0.12, name.replace(" (", "\n("), ha="right", va="top",
                rotation=38, fontsize=8.5)

    ax.annotate("", xy=(1.05, -0.42), xytext=(0.0, -0.42),
                arrowprops=dict(arrowstyle="<->", lw=1.4))
    ax.text(0.5, -0.52, "bending span axis", ha="center", va="top", fontsize=9)

    ax.set_xlim(-0.15, 3.75)
    ax.set_ylim(-1.5, THICKNESS_MM + 0.55)
    ax.set_ylabel("Through-thickness position (mm)")
    ax.set_xticks([])
    ax.set_yticks(np.arange(0, THICKNESS_MM + 0.01, 1.0))
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.minorticks_on()
    ax.tick_params(axis="x", which="both", bottom=False)

    fs.save(fig, "Reply_R4_map_coverage", outdir)
    for name, extent, _ in windows:
        print(f"  {name:36s} {extent:.3f} mm  = {extent / ply:.2f} plies")


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
