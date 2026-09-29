"""The supplementary animation: the depth-resolved map stepped through depth.

Sixteen frames from z = 80 to 290 nm, each showing the elastic modulus, the
hardness and the phase angle of every indent read off its trace on the 5 nm
grid of `depth_resolved_curves.csv`. The colour scales are fixed across the
frames, 0-100 GPa, 0-8 GPa and 0-15 degrees, so a change of colour between
frames is a change of the reading and not of the scale. Blank cells are
indents whose trace does not reach that depth or has no valid reading there.

Run: python depth_slices_animation.py <outdir>
Writes: depth_slices_animation.gif
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

FRAMES_NM = [float(5 * round(z / 5)) for z in np.linspace(80, 290, 16)]
PANELS = (("E_GPa", "viridis", (0, 100), "E (GPa)"),
          ("H_GPa", "viridis", (0, 8), "H (GPa)"),
          ("phase_angle_deg", "magma", (0, 15), "Phase angle (°)"))
FRAME_MS = 700


def lattice(pos):
    xs = np.sort(pos.x_um.unique())
    ys = np.sort(pos.y_um.unique())
    ix = pd.Series(np.searchsorted(xs, pos.x_um.values), index=pos.indent_id)
    iy = pd.Series(np.searchsorted(ys, pos.y_um.values), index=pos.indent_id)
    dx, dy = np.diff(xs).mean(), np.diff(ys).mean()
    extent = (xs.min() - dx / 2, xs.max() + dx / 2,
              ys.min() - dy / 2, ys.max() + dy / 2)
    return ix, iy, (len(ys), len(xs)), extent


def frame(at, ix, iy, shape, extent, z):
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(1, 3, figsize=(12.65, 9.96), dpi=100,
                             layout="constrained")
    for ax, (col, cmap, lim, label) in zip(axes, PANELS):
        arr = np.full(shape, np.nan)
        arr[iy[at.indent_id].values, ix[at.indent_id].values] = at[col].values
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("white")
        im = ax.imshow(arr, origin="lower", extent=extent, cmap=cm,
                       norm=Normalize(*lim), interpolation="nearest")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        fig.colorbar(im, ax=ax, shrink=0.35, label=label)
    fig.suptitle(f"Depth-resolved nanoindentation map      z = {z:.0f} nm",
                 fontweight="bold")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def build(outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    pos = pd.read_csv(DATA / "classified_depth_resolved.csv")[
        ["indent_id", "x_um", "y_um"]]
    curves = pd.read_csv(DATA / "depth_resolved_curves.csv")
    ix, iy, shape, extent = lattice(pos)
    frames = []
    for z in FRAMES_NM:
        at = curves[np.isclose(curves.depth_nm, z)]
        at = at[at.indent_id.isin(pos.indent_id)]
        frames.append(frame(at, ix, iy, shape, extent, z))
    out = Path(outdir) / "depth_slices_animation.gif"
    pal = [f.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
           for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:],
                duration=FRAME_MS, loop=0, optimize=False)
    print(f"  {len(frames)} frames, z = {FRAMES_NM[0]:.0f} to "
          f"{FRAMES_NM[-1]:.0f} nm -> {out.name}")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
