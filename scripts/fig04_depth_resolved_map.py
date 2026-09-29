"""Figure 4 - the depth-resolved map at three depths.

Three panels, one property each: (a) elastic modulus, (b) hardness and (c)
phase angle. Each panel stacks three landscape strips, the slice values at
z = 100, 150 and 200 nm from `classified_depth_resolved.csv`, with the long
axis of the map (y, through the laminate thickness) horizontal. The colour
scales are the ones Figure 3 uses for the load-controlled map, 0-100 GPa and
0-8 GPa, so the two maps read against each other, and 0-15 degrees for the
phase angle. An indent with no valid window at a depth is left blank; values
beyond the scale saturate at its end.

Run: python fig04_depth_resolved_map.py <outdir>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle as fs                                           # noqa: E402

DATA = HERE.parent / "data"
DEPTHS = (100, 150, 200)
PANELS = (("E", "viridis", (0, 100), fs.AXIS["E"], [0, 20, 40, 60, 80, 100]),
          ("H", "magma", (0, 8), fs.AXIS["H"], [0, 2, 4, 6, 8]),
          ("PA", "cividis", (0, 15), fs.AXIS["PA"], [0, 5, 10, 15]))


def grid(df, col):
    """The indents on their acquisition lattice, x along rows, y along columns."""
    xs = np.sort(df.x_um.unique())
    ys = np.sort(df.y_um.unique())
    out = np.full((len(xs), len(ys)), np.nan)
    out[np.searchsorted(xs, df.x_um.values),
        np.searchsorted(ys, df.y_um.values)] = df[col].values
    dx, dy = np.diff(xs).mean(), np.diff(ys).mean()
    extent = (ys.min() - dy / 2, ys.max() + dy / 2,
              xs.min() - dx / 2, xs.max() + dx / 2)
    return out, extent


def build(outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(DATA / "classified_depth_resolved.csv")

    fs.apply()
    W = 6.30
    left, right_cb, cb_w, gap_cb = 0.72, 1.05, 0.14, 0.12
    strip_w = W - left - right_cb - gap_cb - cb_w
    strip_h_in = strip_w * 57 / 297 * 1.05
    inner, outer, top, bottom = 0.04, 0.30, 0.30, 0.55
    H = top + 3 * (3 * strip_h_in + 2 * inner) + 2 * outer + bottom
    fig = plt.figure(figsize=(W, H))
    x0 = left / W

    for p, (key, cmap, lim, label, ticks) in enumerate(PANELS):
        axes = []
        for k, z in enumerate(DEPTHS):
            top_in = top + p * (3 * strip_h_in + 2 * inner + outer) \
                + k * (strip_h_in + inner)
            ax = fig.add_axes([x0, 1 - (top_in + strip_h_in) / H,
                               strip_w / W, strip_h_in / H])
            arr, ext = grid(d, f"{key}_{z}")
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad("white")
            im = ax.imshow(arr, origin="lower", extent=ext, aspect="auto",
                           cmap=cm, norm=Normalize(*lim),
                           interpolation="nearest")
            ax.set_ylabel("x (µm)")
            ax.set_yticks([0, 30])
            if p == len(PANELS) - 1 and k == len(DEPTHS) - 1:
                ax.set_xlabel("y (µm)")
            else:
                ax.set_xticklabels([])
            ax.text(0.985, 0.90, f"z = {z} nm", transform=ax.transAxes,
                    ha="right", va="top", color="white", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.25", fc="0.15",
                              ec="none", alpha=0.85))
            axes.append(ax)
        # one colour bar per panel, spanning its three strips
        b0 = axes[-1].get_position()
        t0 = axes[0].get_position()
        cax = fig.add_axes([(left + strip_w + gap_cb) / W, b0.y0,
                            cb_w / W, t0.y1 - b0.y0])
        cb = fig.colorbar(im, cax=cax, ticks=ticks)
        cb.set_label(label)
        fs.panel_label(axes[0], "abc"[p], dx=-0.10, dy=1.12)

    fs.save(fig, "Fig04_depth_resolved_map", outdir)
    print(f"  {len(d)} indents; blanks at 100/150/200 nm: "
          + ", ".join(str(int(d[f'E_{z}'].isna().sum())) for z in DEPTHS))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
