"""Build the classified depth-resolved table from the primary measurements.

The record and the product are kept apart. `depth_resolved_indents.csv` holds
the primary quantities, the indent coordinates and the strain-rate index among
them, and is declared as a measurement input. `classified_depth_resolved.csv`
carries the zone labels, the quality mask, the constituent class and the
cluster index, which are all decisions taken about those measurements. This
script derives every decision column from the measurement input, so every
column below can be rebuilt and checked:

    zone          which architectural band the indent sits in, from y
    qc_keep       the phase-angle quality decision
    constituent   the four-class rule of classify.py, applied to E at 150 nm
    kmeans_k4     the unsupervised partition, for comparison only

Run: python make_classified_depth.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE.parent / "data"

import classify as C                                           # noqa: E402

# The depth-resolved strip crosses three bands: a flax-rich tow, the basalt
# tow at its centre, and a second flax-rich tow. The boundaries are placed in
# the one-pitch gaps between the bands, which is why they fall on half values.
BANDS = (("flax-bottom", -np.inf, 79.5),
         ("basalt-centre", 79.5, 169.5),
         ("flax-top", 169.5, np.inf))


def zones(y_um):
    out = pd.Series(index=range(len(y_um)), dtype=object)
    for name, lo, hi in BANDS:
        out[(np.asarray(y_um) > lo) & (np.asarray(y_um) <= hi)] = name
    return out.to_numpy()


def main():
    d = pd.read_csv(DATA / "depth_resolved_indents.csv")
    d["zone"] = zones(d.y_um.values)

    # The quality decision, stated as it is applied. The corrected phase angle
    # is tested at the three analysis depths, on the same averaging windows the
    # slice values use, and its magnitude is what counts: a large negative
    # phase angle is as unphysical as a large positive one. An indent whose
    # trace does not cover one of the three depths has no value to test there
    # and is rejected rather than passed by default.
    pa = d[["PA_100", "PA_150", "PA_200"]]
    rejected = ((pa.abs() > C.PA_QC).any(axis=1)) | pa.isna().any(axis=1)
    # The phase-angle rule of 2.2, applied here and stored as a flag.
    d["qc_flag_strict"] = rejected.astype(int)

    band = d.zone.astype(str)
    two = np.where(band.str.contains("basalt"), "basalt", "flax")
    d["constituent"] = C.classify(d.E_150.values, two,
                                  e_upper=C.E_UPPER_DEPTH).values
    d.loc[rejected, "constituent"] = "QC/invalid"

    # The unsupervised partition, for comparison only. -1 marks the indents the
    # quality criterion rejects, which is also what the clustering is not run
    # on, so the two can never disagree about which indents were used.
    import clustering_check as CC
    keep = d.qc_flag_strict == 0
    result = CC.run(d[keep])
    d["kmeans_k4"] = -1
    d.loc[keep, "kmeans_k4"] = np.asarray(result["labels"])
    print(f"  k-means ARI against the rule {result['ARI']:.4f}, "
          f"ARI* {result['ARI_star']:.3f}")

    out = DATA / "classified_depth_resolved.csv"
    cols = ["indent_id", "x_um", "y_um",
            "E_100", "E_150", "E_200", "H_100", "H_150", "H_200",
            "PA_100", "PA_150", "PA_200", "SRS",
            "zone", "qc_flag_strict", "constituent", "kmeans_k4"]
    keep = [c for c in cols if c in d]

    previous = pd.read_csv(out) if out.exists() else None
    d[keep].to_csv(out, index=False, lineterminator="\n")

    print(f"  {len(d)} indents, {int(rejected.sum())} rejected "
          f"({100 * rejected.mean():.1f} %)")
    print(d.constituent.value_counts().to_string())
    b = d.loc[(~rejected) & (d.constituent == "basalt"), "E_150"]
    print(f"\n  basalt median at 150 nm: {b.median():.2f} GPa over {len(b)} "
          f"indents")

    if previous is not None and "constituent" in previous:
        merged = previous[["indent_id", "constituent"]].merge(
            d[["indent_id", "constituent"]], on="indent_id",
            suffixes=("_before", "_after"))
        moved = merged[merged.constituent_before != merged.constituent_after]
        print(f"\n  labels changed on {len(moved)} of {len(merged)} indents")
        if len(moved):
            print(moved.groupby(["constituent_before",
                                 "constituent_after"]).size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
