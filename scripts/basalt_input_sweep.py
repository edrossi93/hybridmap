"""What the upper threshold does to the basalt input, not just to the sum.

Supplementary Table S2 sweeps the classification thresholds and reports how
little the laminate figure moves. That is not the same question as how the
basalt input moves: the pure-basalt class is defined as the tail above the upper
cut, so its median is the median of that tail and must move with the cut even
where the laminate sum does not. That matters because the agreement between the
measured 86 GPa and single-fibre tensile values is used as supporting evidence,
and a number that is an artefact of where the cut was placed cannot support
anything.

So this reports, across the same grid, the basalt input itself: the
pure-basalt median and the number of indents behind it, for the shallow map and
for the depth-resolved one, beside the laminate figure at both zone weightings.

A third block reads the basalt input at a different depth rather than above a
different cut. The contacts of the basalt-centre band that read above 70 GPa
at 80 nm, the population Supplementary Figure S7 follows and
`depth_softening.py` computes, have a median of 84.7 GPa at 80 nm and
97.9 GPa at 50 nm, the shallowest point of the deposited grid. Each is substituted for the basalt input in the two-level
construction with everything else held, exactly as Supplementary Table S4
substitutes the depth-resolved class median, so that the table can carry the
reading depth as a sensitivity beside the cut.

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


def reading_depth_rows(shallow, tiles):
    """The estimate with the basalt input read at 80 and at 50 nm.

    The input is the median of the basalt-centre contacts that read above
    70 GPa at 80 nm, taken from `depth_softening.py` so that this file and
    `depth_softening.json` cannot disagree about it. Everything else in the
    construction, the class fractions, the other three medians and the zone
    share, is the reference's. The relative gaps are on the convention of
    Supplementary Table S4, (estimate - measured) / estimate.
    """
    from depth_softening import basalt_readings
    flex = pd.read_csv(DATA / "flexural_moduli.csv")
    measured = {c: float(g.E_D790_toe_compensated_GPa.mean())
                for c, g in flex.groupby("configuration")}
    per_zone, medians, _ = ZS.zone_fractions(shallow)
    w_mosaic = ZS.tile_share()
    readings = basalt_readings()
    rows = []
    for depth_nm, basalt in sorted(readings["median_GPa"].items(), reverse=True):
        swapped = dict(medians)
        swapped["basalt"] = basalt
        estimate = float(ZS.at_share(per_zone, swapped, w_mosaic)[1][0])
        rows.append({"read_at_nm": depth_nm,
                     "basalt_input_GPa": basalt,
                     "n_contacts": readings["n"],
                     "voigt_mosaic_GPa": estimate,
                     **{f"relative_gap_{c}_pct":
                        100 * (estimate - measured[c]) / estimate
                        for c in ("basalt_face", "flax_face")}})
    return pd.DataFrame(rows)


def main():
    raw = pd.read_csv(DATA / "shallow_map.csv")
    tiles = pd.read_csv(DATA / "bse_segmentation_tiles.csv")
    depth = pd.read_csv(DATA / "classified_depth_resolved.csv")
    shallow = pd.read_csv(DATA / "classified_shallow.csv")

    sh = shallow_rows(raw, tiles)
    dr = depth_rows(depth)
    rd = reading_depth_rows(shallow, tiles)

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

    print("\n  the estimate with the basalt input read at a shallower contact: "
          "the basalt-centre contacts\n  above 70 GPa at 80 nm, "
          "substituted for the basalt median with everything else held")
    print(rd.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    out = {"shallow": sh.to_dict(orient="records"),
           "depth_resolved": dr.to_dict(orient="records"),
           "reading_depth": rd.to_dict(orient="records"),
           "note": ("The pure-basalt class is the tail above the upper cut, so "
                    "its median moves with the cut by construction. Reported "
                    "here so that the agreement with single-fibre values can "
                    "be read against the range rather than at one cut."),
           "reading_depth_note": ("The basalt input read from the same "
                                  "traces at 80 nm and at 50 nm, substituted into the "
                                  "two-level construction as Supplementary "
                                  "Table S4 substitutes the depth-resolved "
                                  "class median. The contacts are those of "
                                  "depth_softening.json.")}
    path = DATA / "basalt_input_sweep.json"
    path.write_text(json.dumps(out, indent=2) + "\n", newline="\n")
    print(f"\n  written: {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
