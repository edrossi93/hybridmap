"""
Indent spacing against the Phani-Oliver criterion, indent by indent.

The criterion asks for a pitch of at least ten times the maximum indentation
depth. Both maps are tested against it on the depth each indent actually
reached:

  * load-controlled map: every indent is driven to the same peak load, so the
    depth follows the material. The stored maximum depth of each indent is
    compared with ten times the pitch along each axis, 2.477 um on the short
    axis and 5.096 um on the long one, class by class.
  * depth-resolved map: every indent is driven to the same 300 nm maximum
    depth on a 3 um grid, which meets the criterion exactly.

The pitch of the load-controlled map was chosen against the maximum depths
expected at 0.53 mN. The two stiff classes, which carry most of the estimate,
meet the criterion on both axes; flax and matrix, which reach about 350 nm,
breach it on most indents along the short axis and on a minority along the
long one. The border test of `border_effect.py` looks for the trace such a
shortfall would leave.

The classified indents carry no quality filter, so failed contacts stay in the
counts: a contact that ran several micrometres deep breaches the criterion like
any other deep contact.

Run: python spacing_compliance.py
Writes: data/spacing_compliance.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

PITCH_NM = {"short": 2477.0, "long": 5096.0}   # load-controlled grid
DEPTH_RESOLVED = {"pitch_nm": 3000.0, "max_depth_nm": 300.0}
CRITERION = 10            # Phani and Oliver: pitch at least ten maximum depths
CLASSES = ["basalt", "mixed-volume", "flax", "matrix"]


def breaching_pct(frame, cls, axis):
    """Share of a class whose maximum depth breaches the criterion on an axis."""
    s = frame.loc[frame.constituent == cls, "max_depth_nm"]
    return 100.0 * float((CRITERION * s > PITCH_NM[axis]).mean())


def main():
    sh = pd.read_csv(DATA / "classified_shallow.csv")
    d = sh[sh.constituent.isin(CLASSES)]
    rows = []
    for c in CLASSES:
        s = d.loc[d.constituent == c, "max_depth_nm"]
        rows.append({
            "class": c, "n": int(len(s)),
            "median_max_depth_nm": round(float(s.median()), 1),
            "required_pitch_at_median_um":
                round(CRITERION * float(s.median()) / 1000.0, 2),
            "breaching_short_axis_pct": round(breaching_pct(d, c, "short"), 1),
            "breaching_long_axis_pct": round(breaching_pct(d, c, "long"), 1)})
    dr_ratio = DEPTH_RESOLVED["pitch_nm"] / DEPTH_RESOLVED["max_depth_nm"]

    print("load-controlled map, pitch 2.48 um (short axis) and 5.10 um (long "
          "axis), maximum depth per indent\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\ndepth-resolved map: pitch / maximum depth = {dr_ratio:g}, the "
          f"criterion is {CRITERION}")

    path = DATA / "spacing_compliance.json"
    path.write_text(json.dumps({
        "criterion": f"pitch >= {CRITERION} x maximum depth",
        "load_controlled_pitch_nm": PITCH_NM,
        "classified_indents": int(len(d)),
        "rows": rows,
        "depth_resolved": {**DEPTH_RESOLVED,
                           "pitch_over_max_depth": dr_ratio}},
        indent=2) + "\n", encoding="utf8", newline="\n")
    print(f"\nwritten to {path.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
