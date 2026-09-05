"""Do indents on the edge of the grid read differently from those inside it?

The indent spacing on the short axis of the fixed-depth map is below the ten
contact depths that Phani and Oliver ask for, and the paper says so. The
question that follows is whether the shortfall shows up in the data. If adjacent
plastic zones or elastic fields were interacting, an indent on the outer row of
the grid would feel it less than one surrounded on both sides, so the two
populations of the same class in the same zone would differ.

The border is defined on the short axis alone, the outermost row at each end,
because that is the axis whose pitch breaches the criterion; on the long axis the
pitch satisfies it for every class and there is nothing to test. Classes are
compared within their own architectural zone, since a class can sit in a
different neighbourhood on either side of the zone boundary.

The test is a two-sided Mann-Whitney on the indentation modulus, which makes no
assumption about the shape of the distributions, and the five comparisons carry a
Holm correction, without which the smallest of five p values means little.

Run: python border_effect.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu

DATA = Path(__file__).resolve().parent.parent / "data"
MIN_N = 10                      # below this the test says nothing either way


def main():
    d = pd.read_csv(DATA / "classified_shallow.csv")
    d = d[d.constituent != "QC/invalid"]

    y = np.sort(d.y_um.unique())
    d = d.assign(border=(d.y_um == y.min()) | (d.y_um == y.max()))
    print(f"short-axis pitch {np.diff(y).mean() * 1000:.0f} nm, "
          f"{len(y)} rows, {int(d.border.sum())} indents on the two outer rows")

    rows = []
    for (zone, cls), g in d.groupby(["zone", "constituent"]):
        b, i = g[g.border], g[~g.border]
        if len(b) < MIN_N or len(i) < MIN_N:
            continue
        p = mannwhitneyu(b.E_GPa, i.E_GPa, alternative="two-sided").pvalue
        rows.append((zone, cls, len(b), len(i),
                     b.E_GPa.median(), i.E_GPa.median(), p))

    # Holm: sort by p, and each is compared against alpha / (m - rank).
    rows.sort(key=lambda r: r[-1])
    m = len(rows)
    print(f"\n{'zone':14s} {'class':13s} {'n border':>8s} {'n inside':>8s} "
          f"{'E border':>9s} {'E inside':>9s} {'p':>7s} {'Holm p':>8s}")
    running = 0.0
    for k, (zone, cls, nb, ni, eb, ei, p) in enumerate(rows):
        running = max(running, min(1.0, p * (m - k)))
        print(f"{zone:14s} {cls:13s} {nb:8d} {ni:8d} {eb:9.2f} {ei:9.2f} "
              f"{p:7.3f} {running:8.3f}")

    print("\nAn interaction between neighbouring indents would make the border "
          "population\nthe stiffer of the two, since it has fewer neighbours. "
          "Any comparison below\nthat runs the other way is evidence against "
          "the interaction, not for it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
