"""Are the quality rejections of the depth-resolved map an edge artefact?

The Methods report that the indents the phase-angle criterion rejects are spread
across the mapped area and are no more frequent at its border than inside it.
This script makes that comparison.

The border is the outer row at each end of the short axis, which is how
`border_effect.py` defines it for the load-controlled map, and for the same
reason: the short axis is the one whose pitch is judged against the Phani and
Oliver criterion. On this map the two pitches are equal, 3.0 um each, so the
short axis is the shorter side rather than the slacker one, and the long axis is
reported below as a check that the answer does not depend on which side is
chosen.

An edge artefact would concentrate the rejections on the outer rows: a contact
established at the edge of the scanned field has a different thermal and drift
history from one in the middle of it. The comparison is a rate against a rate,
with the counts printed beside them, because two rates within a few tenths of a
point of each other say nothing without the number of indents behind them.

What this cannot do. It does not test the cause of any difference it finds, and
a null result here is evidence against an edge artefact rather than proof of its
absence: 200 indents on the two outer rows put a floor under how small a
difference could be resolved.

Run: python border_rejection.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def rejection_rates(depth, axis="x_um"):
    """Rejected fraction on the two outer rows of one axis, and inside them.

    Returns a dict with both rates in per cent and the counts behind each.
    """
    coordinate = np.sort(depth[axis].unique())
    border = depth[axis].isin([coordinate.min(), coordinate.max()])
    out = {}
    for name, sub in (("border", depth[border]), ("interior", depth[~border])):
        rejected = int(sub.qc_flag_strict.sum())
        out[name] = {"n": len(sub), "rejected": rejected,
                     "pct": 100.0 * rejected / len(sub)}
    out["axis"] = axis
    out["rows"] = len(coordinate)
    out["pitch_um"] = float(np.diff(coordinate).mean())
    return out


def main():
    depth = pd.read_csv(DATA / "classified_depth_resolved.csv")
    total = 100.0 * depth.qc_flag_strict.mean()
    print(f"{len(depth)} indents, {int(depth.qc_flag_strict.sum())} rejected "
          f"({total:.2f} %)\n")

    # The short axis first, because that is the comparison the Methods quote.
    for axis, label in (("x_um", "short axis, in plane"),
                        ("y_um", "long axis, through the thickness")):
        r = rejection_rates(depth, axis)
        print(f"{label}: {r['rows']} rows at {r['pitch_um']:.2f} um")
        for name in ("border", "interior"):
            d = r[name]
            print(f"  {name:<9} {d['rejected']:>4} of {d['n']:>4} rejected, "
                  f"{d['pct']:.2f} %")
        print(f"  difference {r['border']['pct'] - r['interior']['pct']:+.2f} "
              f"points\n")

    # Where the rejections do concentrate, which is the sentence the border
    # comparison sits beside: they follow the local stiffness, not the edges.
    print("by architectural band")
    for zone, g in depth.groupby("zone"):
        print(f"  {zone:<14} {int(g.qc_flag_strict.sum()):>4} of {len(g):>4} "
              f"rejected, {100 * g.qc_flag_strict.mean():.1f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
