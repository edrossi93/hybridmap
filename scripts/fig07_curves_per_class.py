"""Figure 7 - loading curves and depth signatures, per mechanical class.

Rebuilt from the deposited curve file, matching the submitted figure element for
element:

  row (a) one panel per class. Thin coloured lines are individual load-depth
      records, a fixed random subsample so the panel stays readable; the heavy
      black line is the median load at each depth over the whole class, which is
      the quantity the text refers to. A boxed label gives the class and its n.
  row (b) median and +/- 1 sigma band of elastic modulus, hardness and phase
      angle against indentation depth, all four classes on shared axes.

The sigma band is retained here. Reviewer 3 comment 22 objects to a +/- 1 sigma
envelope drawn around a single specimen, which is Figure 9; over a population of
hundreds of indents it is a legitimate summary.

Axes are placed explicitly, so there is room outside each panel for its letter.
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
GRID = np.arange(70.0, 300.5, 5.0)
N_SHOWN = 30          # individual curves drawn per panel
SEED = 42


def resample(curves, ids, column, grid=GRID):
    """Every curve on a common depth grid; the acquisition does not sample
    identical depths from indent to indent."""
    out = []
    for _, g in curves[curves.indent_id.isin(ids)].groupby("indent_id"):
        g = g.sort_values("depth_nm")
        d, v = g.depth_nm.to_numpy(), g[column].to_numpy()
        ok = np.isfinite(d) & np.isfinite(v)
        if ok.sum() >= 10:
            out.append(np.interp(grid, d[ok], v[ok], left=np.nan, right=np.nan))
    return np.vstack(out) if out else None


def build(outdir):
    # Created here rather than assumed. Run from a fresh checkout with an
    # output directory that does not exist yet, these scripts used to fail
    # on the savefig at the end, after doing all the work.
    Path(outdir).mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(DATA / "classified_depth_resolved.csv")
    cur = pd.read_csv(DATA / "depth_resolved_curves.csv")

    fs.set_style("bold", font_scale=0.58)

    W = 6.10
    LEFT, GAP_A, GAP_B, RIGHT, TOP = 0.52, 0.34, 0.50, 0.18, 0.30
    ROW_H, ROW_GAP, BOTTOM = 1.28, 0.72, 0.60
    AW = (W - LEFT - 3 * GAP_A - RIGHT) / 4
    BW = (W - LEFT - 2 * GAP_B - RIGHT) / 3
    H = TOP + ROW_H + ROW_GAP + ROW_H + BOTTOM
    fig = plt.figure(figsize=(W, H))

    def place(x_in, w_in, y_top_in):
        return fig.add_axes([x_in / W, 1 - (y_top_in + ROW_H) / H,
                             w_in / W, ROW_H / H])

    # ---- row (a): one panel per class -------------------------------------
    rng = np.random.default_rng(SEED)
    row_a = []
    for i, cls in enumerate(ORDER):
        ax = place(LEFT + i * (AW + GAP_A), AW, TOP)
        row_a.append(ax)
        ids = lab.loc[lab.constituent == cls, "indent_id"].to_numpy()
        shown = rng.choice(ids, size=min(N_SHOWN, len(ids)), replace=False)
        for iid in shown:
            g = cur[cur.indent_id == iid].sort_values("depth_nm")
            ax.plot(g.depth_nm, g.load_mN, color=fs.CONSTITUENT[cls],
                    lw=0.45, alpha=0.55, zorder=2)
        stack = resample(cur, ids, "load_mN")
        ax.plot(GRID, np.nanmedian(stack, axis=0), color="k", lw=1.8, zorder=4)
        ax.set_xlim(50, 350)
        ax.set_ylim(0, 2.0)
        ax.set_xlabel(fs.AXIS["depth"])
        # every panel keeps its own y tick labels, as in the submitted figure;
        # only the axis label is carried by the first
        if i == 0:
            ax.set_ylabel(fs.AXIS["load"])
        # the flax and matrix curves stay low, so the top left is clear there;
        # the two basalt classes climb steeply and leave the bottom right clear
        top_left = cls in ("flax", "matrix")
        ax.text(0.03 if top_left else 0.97, 0.97 if top_left else 0.03,
                f"{LABEL[cls]}\n(n={len(ids)})", transform=ax.transAxes,
                ha="left" if top_left else "right",
                va="top" if top_left else "bottom", fontsize=5.6,
                bbox=dict(boxstyle="round,pad=0.25", fc="w", ec="0.4", lw=0.6))
        ax.grid(True, lw=0.4, color="0.9", zorder=0)

    # ---- row (b): population profiles -------------------------------------
    row_b = []
    y_top_b = TOP + ROW_H + ROW_GAP
    for i, (col, ylab, ylim) in enumerate(
            (("E_GPa", fs.AXIS["E"], (0, 75)),
             ("H_GPa", fs.AXIS["H"], (0, 5)),
             ("phase_angle_deg", fs.AXIS["PA"], (0, 10)))):
        ax = place(LEFT + i * (BW + GAP_B), BW, y_top_b)
        row_b.append(ax)
        for cls in ORDER:
            ids = lab.loc[lab.constituent == cls, "indent_id"]
            stack = resample(cur, ids, col)
            if stack is None:
                continue
            med, sd = np.nanmedian(stack, axis=0), np.nanstd(stack, axis=0)
            c = fs.CONSTITUENT[cls]
            ax.fill_between(GRID, med - sd, med + sd, color=c, alpha=0.22, lw=0)
            ax.plot(GRID, med, color=c, lw=1.6,
                    label=f"{LABEL[cls]} (n={len(ids)})")
        ax.set_xlim(70, 300)
        ax.set_ylim(*ylim)
        ax.set_xlabel(fs.AXIS["depth"])
        ax.set_ylabel(ylab)
        ax.grid(True, lw=0.4, color="0.9", zorder=0)
    row_b[0].legend(loc="upper right", fontsize=5.2, frameon=False,
                    handlelength=1.1, labelspacing=0.3, borderpad=0.15)

    for ax in row_a + row_b:
        ax.minorticks_on()

    # (a) labels the first panel of the top row, (b) the first of the bottom
    fs.panel_letters(fig, [row_a[0], row_b[0]], letters=["a", "b"], fontsize=11,
                     where="above")

    report = fs.layout_report(fig)
    for key, hits in report.items():
        if hits:
            print(f"  layout: {key}: {hits}")

    fig.savefig(Path(outdir) / "Fig07_loading_curves_depth_signatures.pdf")
    fig.savefig(Path(outdir) / "Fig07_loading_curves_depth_signatures.png", dpi=400)
    plt.close(fig)
    print(f"  wrote Figure_7, {W:.2f} x {H:.2f} in, embed at exactly {W:.2f} in")
    for cls in ORDER:
        ids = lab.loc[lab.constituent == cls, "indent_id"]
        st = resample(cur, ids, "load_mN")
        print(f"    {LABEL[cls]:13s} n={len(ids):5d}  median load at 300 nm "
              f"{np.nanmedian(st, axis=0)[-1]:6.3f} mN")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
