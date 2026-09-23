"""What the upper threshold does to the basalt input, not just to the sum.

Supplementary Table S2 sweeps the classification thresholds and reports how
little the laminate figure moves. Reviewer 3 points out that this is not the
question he asked: the pure-basalt class is defined as the tail above the upper
cut, so its median is the median of that tail and must move with the cut even
where the laminate sum does not. That matters because the agreement between the
measured 86 GPa and single-fibre tensile values is used as supporting evidence,
and a number that is an artefact of where the cut was placed cannot support
anything.

So this reports, across the same grid, the quantity he asked for: the
pure-basalt median and the number of indents behind it, for the shallow map and
for the depth-resolved one, beside the laminate figure at both zone weightings.

Run: python basalt_input_sweep.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE.parent / "data"

import classify as C                                            # noqa: E402
import zone_share as ZS                                         # noqa: E402

LOWER = (10, 15, 20)
UPPER_SHALLOW = (60, 65, 70, 75, 80)
UPPER_DEPTH = (30, 35, 40, 45, 50)


def shallow_rows(raw, tiles):
    d = raw[raw.x_um >= C.X_RESIN].copy()
    d["zone"] = C.assign_zone(d.x_um.values).values
    w_mosaic = tiles.n_basaltzone.sum() / (tiles.n_basaltzone.sum()
                                           + tiles.n_flaxzone.sum())
    rows = []
    for lower in LOWER:
        for upper in UPPER_SHALLOW:
            f = d.assign(constituent=C.classify(
                d.E_GPa.values, d.zone.values, lower, upper).values)
            basalt = f.loc[f.constituent == "basalt", "E_GPa"]
            per_zone, medians, w_ind = ZS.zone_fractions(f)
            row = {"lower_GPa": lower, "upper_GPa": upper,
                   "basalt_median_GPa": float(basalt.median()),
                   "n_basalt": int(len(basalt))}
            for tag, w in (("window", w_ind), ("mosaic", w_mosaic)):
                row[f"voigt_{tag}_GPa"] = float(
                    ZS.at_share(per_zone, medians, w)[1][0])
            rows.append(row)
    return pd.DataFrame(rows)


def depth_rows(d):
    """The same question on the depth-resolved map, where the cut is 40 GPa."""
    keep = d[d.qc_flag_strict == 0]
    rows = []
    for upper in UPPER_DEPTH:
        basalt = keep.loc[keep.E_150 > upper, "E_150"]
        rows.append({"upper_GPa": upper,
                     "basalt_median_GPa": float(basalt.median()),
                     "n_basalt": int(len(basalt))})
    return pd.DataFrame(rows)


def main():
    raw = pd.read_csv(DATA / "shallow_map.csv")
    tiles = pd.read_csv(DATA / "bse_segmentation_tiles.csv")
    depth = pd.read_csv(DATA / "classified_depth_resolved.csv")

    sh = shallow_rows(raw, tiles)
    dr = depth_rows(depth)

    print("  shallow map, pure-basalt class")
    print(sh.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    at70 = sh[(sh.lower_GPa == 15) & (sh.upper_GPa == 70)].iloc[0]
    span = sh.basalt_median_GPa.max() - sh.basalt_median_GPa.min()
    print(f"\n  at the paper's cuts: {at70.basalt_median_GPa:.2f} GPa over "
          f"{int(at70.n_basalt)} indents")
    print(f"  across the grid the basalt median spans "
          f"{sh.basalt_median_GPa.min():.2f} to "
          f"{sh.basalt_median_GPa.max():.2f} GPa, a range of {span:.2f}, "
          f"while the laminate figure moves "
          f"{sh.voigt_mosaic_GPa.max() - sh.voigt_mosaic_GPa.min():.2f}")

    print("\n  depth-resolved map, pure-basalt class")
    print(dr.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    out = {"shallow": sh.to_dict(orient="records"),
           "depth_resolved": dr.to_dict(orient="records"),
           "note": ("The pure-basalt class is the tail above the upper cut, so "
                    "its median moves with the cut by construction. Reported "
                    "here so that the agreement with single-fibre values can "
                    "be read against the range rather than at one cut.")}
    path = DATA / "basalt_input_sweep.json"
    path.write_text(json.dumps(out, indent=2) + "\n", newline="\n")
    print(f"\n  written: {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
