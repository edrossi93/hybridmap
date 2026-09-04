"""
Sensitivity of the depth-resolved analysis to the phase-angle QC threshold.

Reviewer 3 asked for this and it was never run: "Please add a sensitivity
analysis on the phase-angle QC threshold (10° / 15° / 20°) to Tables S1-S3.
Rejecting 22 % of indents is a substantial fraction."

He is right that the fraction is substantial and right that a threshold which
discards a fifth of the data should be shown not to be doing the work. The
criterion identifies indents whose contact never left the near-surface
transient, where the measured stiffness belongs to the establishing contact
rather than to the material. Lowering the threshold rejects more of them and
also more good data; raising it keeps more of both.

What the sweep changes and what it cannot. The threshold applies only to the
depth-resolved acquisition, which is the one carrying a continuous-stiffness
trace: the fixed-depth map has no phase-angle channel, so the reference Voigt
bound of 21.33 GPa, built from the fixed-depth medians, is untouched by any
choice made here. What moves is the depth-resolved class medians, and with them
the alternative basalt input of Supplementary Table S2 and the bound that input
produces. Saying so is part of the answer: the threshold cannot influence the
headline number, and the sweep shows how little it moves the one number it can.

The near-surface limit at z = 100 nm is swept with the main threshold, five
degrees above it, keeping the shape of the original criterion rather than
freezing one half of it while the other moves.

One honest caveat, and it has to be stated wherever these numbers appear. The
published flag was computed on the full continuous-stiffness trace and rejects
433 of the 2000 indents. The deposit carries the trace only at the three slice
depths, so recomputing the same criterion here at the same 15 degrees rejects
320. The extra 113 are indents whose phase angle exceeds the limit somewhere
between the slices, and they are on average more compliant, which is why the
depth-resolved basalt median sits at 49.47 GPa under the published flag and at
52.76 GPa under the recomputed one.

The sweep is therefore internally consistent, the same rule applied at three
thresholds to the same data, and it is not an attempt to reproduce the published
counts. What it answers is the reviewer's question: whether the choice of
threshold carries the result. The published baseline is reported alongside so
that the two can be told apart at a glance.
"""

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE.parent / "data"

from classify import classify, phase_angle_qc, E_UPPER_DEPTH   # noqa: E402
from rule_of_mixtures import bounds, class_table, MEDIAN_COL   # noqa: E402

THRESHOLDS = (10.0, 15.0, 20.0)
CLASSES = ["flax", "matrix", "mixed-volume", "basalt"]


def relabel(df, limit):
    d = df.copy()
    band = d["zone"].astype(str)
    zone = ["basalt" if "basalt" in b else "flax" for b in band]
    d["constituent"] = classify(d.E_150.values, zone,
                                e_upper=E_UPPER_DEPTH).values
    keep = phase_angle_qc(d.PA_100, d.PA_150, d.PA_200,
                          limit=limit, near_surface_limit=limit + 5.0)
    d.loc[~keep, "constituent"] = "QC/invalid"
    return d


def main():
    depth = pd.read_csv(DATA / "classified_depth_resolved.csv")
    shallow = pd.read_csv(DATA / "classified_shallow.csv")
    table = class_table(shallow)
    reference = bounds(table.fraction, table[MEDIAN_COL])[0]

    rows = []
    # The published baseline, on the stored flag, so the recomputed sweep can be
    # read against the numbers the paper actually reports.
    stored = depth.copy()
    med0 = {c: float(stored.loc[stored.constituent == c, "E_150"].median())
            for c in CLASSES}
    alt0 = table.copy()
    alt0.loc[alt0["class"] == "basalt", MEDIAN_COL] = med0["basalt"]
    rows.append({
        "phase_angle_limit_deg": "15 (published flag, full trace)",
        "rejected": int((stored.constituent == "QC/invalid").sum()),
        "rejected_pct": 100.0 * (stored.constituent == "QC/invalid").sum() / len(stored),
        **{f"E_{c}_GPa": round(med0[c], 2) for c in CLASSES},
        "alternative_basalt_input_GPa": round(med0["basalt"], 2),
        "Voigt_with_that_input_GPa":
            round(float(bounds(alt0.fraction, alt0[MEDIAN_COL])[0]), 2),
    })

    for limit in THRESHOLDS:
        d = relabel(depth, limit)
        rejected = (d.constituent == "QC/invalid").sum()
        med = {c: float(d.loc[d.constituent == c, "E_150"].median())
               for c in CLASSES}
        # The one number the threshold can move: the alternative basalt input
        # and the bound it gives when substituted into the reference table.
        alt = table.copy()
        alt.loc[alt["class"] == "basalt", MEDIAN_COL] = med["basalt"]
        rows.append({
            "phase_angle_limit_deg": limit,
            "rejected": int(rejected),
            "rejected_pct": 100.0 * rejected / len(d),
            **{f"E_{c}_GPa": round(med[c], 2) for c in CLASSES},
            "alternative_basalt_input_GPa": round(med["basalt"], 2),
            "Voigt_with_that_input_GPa":
                round(float(bounds(alt.fraction, alt[MEDIAN_COL])[0]), 2),
        })

    out = pd.DataFrame(rows)
    print(f"reference Voigt bound, fixed-depth inputs: {reference:.2f} GPa "
          f"(unaffected: the fixed-depth map has no phase-angle channel)\n")
    print(out.to_string(index=False))

    spread = out["Voigt_with_that_input_GPa"].iloc[1:]
    print(f"\nacross the recomputed sweep the alternative bound moves "
          f"{spread.min():.2f} to {spread.max():.2f} GPa, "
          f"a range of {spread.max() - spread.min():.2f} GPa; "
          f"the rejected fraction moves "
          f"{out.rejected_pct.min():.0f} % to {out.rejected_pct.max():.0f} %")

    path = DATA / "phase_angle_sensitivity.json"
    path.write_text(json.dumps({
        "reference_voigt_GPa": reference,
        "note": ("The threshold applies only to the depth-resolved "
                 "acquisition. The reference bound is built from the "
                 "fixed-depth map, which has no phase-angle channel, so no "
                 "choice of threshold can move it."),
        "rows": rows}, indent=2) + "\n", encoding="utf8", newline="\n")
    print(f"\nwritten to {path.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
