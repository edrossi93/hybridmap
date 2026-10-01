"""Supplementary Figure S7 - how the stiff contacts soften with depth.

Stepping through the depth slices, the positions that sit at the top of the colour scale at about 80 nm fade as the
contact deepens, and if the *same individual positions* soften rather than only
the class median, that establishes that the depth-resolved basalt median is not
a second estimate of the fibre modulus: 86 GPa lies within what these contacts
read between 50 and 100 nm, and their reading falls with depth.

They do. Section 4.4 reads the fall as the surroundings of the fibre entering
the measurement as the load grows: the elastic field extends over several
contact radii, and the fibre is a stiff column held by a weakly bonded
interface in a compliant matrix, within an epoxy mount, all in series with the
contact. The softening does not follow the distance to compliant material at
the surface, which on fibres 10 to 20 um across excludes only a lateral
proximity effect at the scale of the 3 um grid. The panel on the left follows every indent in the basalt-centre zone
that reads above the shallow map's own 70 GPa cut at 80 nm, one line each, from
50 to 300 nm. The panel on the right is the same population as a median with an
interquartile band, beside the class the depth-resolved classification itself
calls basalt.

The selection is made at 80 nm and the behaviour is read afterwards, so it is
not circular: nothing about the trace below 80 nm enters the choice of which
traces to draw.

Run: python figS7_basalt_depth.py <outdir>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle as fs                                           # noqa: E402

DATA = HERE.parent / "data"

SELECT_AT_NM = 80.0
SELECT_ABOVE_GPA = 70.0        # the shallow map's own upper threshold


def build(outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(DATA / "depth_resolved_curves.csv")
    classified = pd.read_csv(DATA / "classified_depth_resolved.csv")

    zone = classified[classified.zone == "basalt-centre"]
    wide = (curves[curves.indent_id.isin(zone.indent_id)]
            .pivot(index="indent_id", columns="depth_nm", values="E_GPa"))
    stiff = wide.index[wide[SELECT_AT_NM] > SELECT_ABOVE_GPA]
    called = zone.loc[zone.constituent == "basalt", "indent_id"]

    fs.set_style("bold", font_scale=0.78)
    fs.use_arial()
    W, H = 6.30, 2.95
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, H), layout="constrained")

    z = wide.columns.to_numpy(float)
    for ind in stiff:
        ax1.plot(z, wide.loc[ind].to_numpy(), lw=0.6, alpha=0.35,
                 color=fs.PAPER["basalt"], zorder=2)
    ax1.plot(z, wide.loc[stiff].median().to_numpy(), lw=2.4, color="k",
             zorder=4, label=f"median of {len(stiff)} indents")
    ax1.set_xlabel("Depth, z (nm)")
    ax1.set_ylabel(fs.AXIS["E"])
    ax1.set_title(f"Reading above {SELECT_ABOVE_GPA:.0f} GPa at "
                  f"{SELECT_AT_NM:.0f} nm", fontsize=8.5)
    ax1.legend(loc="upper right", fontsize=7.5, frameon=False)

    for ids, colour, name in ((stiff, fs.PAPER["basalt"],
                               f"above {SELECT_ABOVE_GPA:.0f} GPa at "
                               f"{SELECT_AT_NM:.0f} nm"),
                              (called, fs.PAPER["mixed-volume"],
                               "class called basalt at 150 nm")):
        block = wide.loc[wide.index.isin(ids)]
        med = block.median().to_numpy()
        lo = block.quantile(0.25).to_numpy()
        hi = block.quantile(0.75).to_numpy()
        ax2.fill_between(z, lo, hi, color=colour, alpha=0.22, lw=0, zorder=2)
        ax2.plot(z, med, lw=2.0, color=colour, zorder=3,
                 label=f"{name} (n = {len(block)})")
    ax2.set_xlabel("Depth, z (nm)")
    ax2.set_ylabel(fs.AXIS["E"])
    ax2.set_title("Median and interquartile range", fontsize=8.5)
    ax2.legend(loc="upper right", fontsize=7.5, frameon=False)

    for ax in (ax1, ax2):
        ax.set_xlim(50, 300)
        ax.minorticks_on()

    fs.panel_letters(fig, [ax1, ax2], where="above")
    fig.savefig(Path(outdir) / "FigS7_basalt_depth_dependence.pdf")
    fig.savefig(Path(outdir) / "FigS7_basalt_depth_dependence.png", dpi=400)
    plt.close(fig)

    ratio = (wide.loc[stiff, 300.0] / wide.loc[stiff, SELECT_AT_NM]).dropna()
    print(f"  {len(stiff)} indents selected above {SELECT_ABOVE_GPA:.0f} GPa "
          f"at {SELECT_AT_NM:.0f} nm, {100 * len(stiff) / len(wide):.0f} % of "
          f"the zone")
    print(f"  median E: {wide.loc[stiff, SELECT_AT_NM].median():.1f} GPa at "
          f"{SELECT_AT_NM:.0f} nm, {wide.loc[stiff, 300.0].median():.1f} GPa "
          f"at 300 nm")
    print(f"  E(300)/E(80): median {ratio.median():.3f}, "
          f"{100 * (ratio < 1).mean():.0f} % of them soften")
    return 0


if __name__ == "__main__":
    sys.exit(build(sys.argv[1] if len(sys.argv) > 1 else "."))
